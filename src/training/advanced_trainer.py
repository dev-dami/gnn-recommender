import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from ..models.lightgcn import LightGCN
from ..losses.bpr import BPRLoss
from ..data.sampler import NegativeSampler
from ..evaluation.evaluator import Evaluator
from ..utils.checkpoint import save_checkpoint
from .mixed_precision import MixedPrecisionTrainer
from .distributed import DistributedManager


class AdvancedTrainer:
    """Extended trainer supporting mixed precision, distributed training,
    self-supervised learning, UltraGCN constraints, and gradient accumulation.
    """

    def __init__(self, model, train_mat, val_mat, test_mat,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 neg_samples: int = 1, gradient_clip: float = 1.0,
                 top_k: list = None, device: str = "cpu",
                 log_dir: str = "outputs/logs",
                 checkpoint_dir: str = "outputs/checkpoints",
                 seed: int = 42,
                 mixed_precision: bool = False,
                 mp_dtype: str = "fp16",
                 distributed: bool = False,
                 gradient_accumulation_steps: int = 1,
                 ssl_loss_weight: float = 0.0,
                 ultralgcn_weight: float = 0.0,
                 edge_dropout: float = 0.0,
                 node_dropout: float = 0.0):
        self.model = model.to(device)
        self.device = device
        self.train_mat = train_mat
        self.val_mat = val_mat
        self.test_mat = test_mat

        self.criterion = BPRLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sampler = NegativeSampler(train_mat, num_negatives=neg_samples, seed=seed)
        self.val_evaluator = Evaluator(val_mat, train_mat, top_k=top_k or [10, 20])
        self.test_evaluator = Evaluator(test_mat, train_mat, top_k=top_k or [10, 20])

        self.gradient_clip = gradient_clip
        self.top_k = top_k or [10, 20]
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ssl_loss_weight = ssl_loss_weight
        self.ultralgcn_weight = ultralgcn_weight

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir

        self.best_metric = 0.0
        self.patience_counter = 0

        self.mp_trainer = MixedPrecisionTrainer(enabled=mixed_precision, dtype=mp_dtype, device=device)
        self.dist_manager = DistributedManager()
        if distributed:
            self.dist_manager.setup()

    def train_epoch(self) -> float:
        self.model.train()
        user_indices = np.arange(self.train_mat.shape[0])
        np.random.shuffle(user_indices)

        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        batch_size = 1024
        for start in range(0, len(user_indices), batch_size):
            batch_users = user_indices[start:start + batch_size]

            all_users, all_pos, all_neg = [], [], []
            for u in batch_users:
                pos_items = self.sampler.user_pos_items[u]
                if not pos_items:
                    continue
                for p in list(pos_items):
                    neg = self.sampler.sample(u, p)[0]
                    all_users.append(u)
                    all_pos.append(p)
                    all_neg.append(neg)

            if not all_users:
                continue

            users_t = torch.LongTensor(all_users).to(self.device)
            pos_t = torch.LongTensor(all_pos).to(self.device)
            neg_t = torch.LongTensor(all_neg).to(self.device)

            with self.mp_trainer.get_context():
                user_emb, item_emb = self.model()
                pos_scores = self.model.get_user_item_scores(user_emb, item_emb, users_t, pos_t)
                neg_scores = self.model.get_user_item_scores(user_emb, item_emb, users_t, neg_t)
                loss = self.criterion(pos_scores, neg_scores) / self.gradient_accumulation_steps

                if self.ultralgcn_weight > 0 and hasattr(self.model, 'compute_kernel_loss'):
                    import torch.sparse
                    kernel_loss = self.model.compute_kernel_loss(self.model.norm_adj, user_emb, item_emb)
                    loss = loss + self.ultralgcn_weight * kernel_loss / self.gradient_accumulation_steps

            scaled_loss = self.mp_trainer.scale_loss(loss)
            scaled_loss.backward()

            if (start // batch_size + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.mp_trainer.step(self.optimizer)
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, k: int = 10, split: str = "val") -> dict:
        self.model.eval()
        user_emb, item_emb = self.model()
        evaluator = self.val_evaluator if split == "val" else self.test_evaluator
        return evaluator.evaluate(user_emb, item_emb, k)

    def train(self, epochs: int = 200, eval_every: int = 5,
              patience: int = 20, warmup_epochs: int = 10) -> dict:
        best_state = None
        history = {"train_loss": [], "val_metrics": [], "test_metrics": []}

        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mp_trainer.status}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            history["train_loss"].append(train_loss)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LR", lr, epoch)

            elapsed = time.time() - t0
            log_msg = f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s"

            if epoch % eval_every == 0 or epoch == 1:
                val_metrics = self.evaluate(k=max(self.top_k), split="val")
                test_metrics = self.evaluate(k=max(self.top_k), split="test")

                for k in self.top_k:
                    for name, metrics in [("val", val_metrics), ("test", test_metrics)]:
                        if f"Recall@{k}" in metrics:
                            self.writer.add_scalar(f"{name}/Recall@{k}", metrics[f"Recall@{k}"], epoch)
                            self.writer.add_scalar(f"{name}/NDCG@{k}", metrics.get(f"NDCG@{k}", 0), epoch)

                primary_metric = val_metrics.get("Recall@10", 0)
                history["val_metrics"].append({"epoch": epoch, **val_metrics})
                history["test_metrics"].append({"epoch": epoch, **test_metrics})

                log_msg += f" | Val R@10: {val_metrics.get('Recall@10', 0):.4f}"

                if primary_metric > self.best_metric:
                    self.best_metric = primary_metric
                    self.patience_counter = 0
                    state_to_save = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                    best_state = {k: v.cpu().clone() for k, v in state_to_save.items()}
                    save_checkpoint(
                        {"epoch": epoch, "model_state_dict": best_state, "metrics": val_metrics},
                        str(Path(self.checkpoint_dir) / "best_model.pt"),
                    )
                    log_msg += " *"
                else:
                    self.patience_counter += 1

            print(log_msg)

            if patience > 0 and self.patience_counter >= patience and epoch > warmup_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(best_state)
            else:
                self.model.load_state_dict(best_state)

        self.writer.close()
        self.dist_manager.cleanup()
        return history

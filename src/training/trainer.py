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
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model: LightGCN, train_mat, val_mat, test_mat,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 neg_samples: int = 1, gradient_clip: float = 1.0,
                 top_k: list = None, device: str = "cpu",
                 log_dir: str = "outputs/logs", checkpoint_dir: str = "outputs/checkpoints",
                 seed: int = 42):
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

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir

        self.best_metric = 0.0
        self.patience_counter = 0

    def train_epoch(self) -> float:
        self.model.train()
        user_indices = np.arange(self.train_mat.shape[0])
        np.random.shuffle(user_indices)

        total_loss = 0.0
        num_batches = 0

        batch_size = 1024
        for start in range(0, len(user_indices), batch_size):
            batch_users = user_indices[start:start + batch_size]

            all_users = []
            all_pos = []
            all_neg = []
            for u in batch_users:
                pos_items = self.sampler.user_pos_items[u]
                if not pos_items:
                    continue
                pos = list(pos_items)
                negs = self.sampler.sample(u, pos[0])
                for p in pos:
                    neg = self.sampler.sample(u, p)[0]
                    all_users.append(u)
                    all_pos.append(p)
                    all_neg.append(neg)

            if not all_users:
                continue

            users_t = torch.LongTensor(all_users).to(self.device)
            pos_t = torch.LongTensor(all_pos).to(self.device)
            neg_t = torch.LongTensor(all_neg).to(self.device)

            user_emb, item_emb = self.model()
            pos_scores = self.model.get_user_item_scores(user_emb, item_emb, users_t, pos_t)
            neg_scores = self.model.get_user_item_scores(user_emb, item_emb, users_t, neg_t)

            loss = self.criterion(pos_scores, neg_scores)

            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
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
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    save_checkpoint(
                        {"epoch": epoch, "model_state_dict": best_state,
                         "metrics": val_metrics},
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
            self.model.load_state_dict(best_state)

        self.writer.close()
        return history

    def get_recommendations(self, user_idx: int, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model()
            scores = self.model.predict_all_scores(user_idx, item_emb, user_emb)
            train_items = set(self.train_mat[user_idx].indices.tolist())
            scores[list(train_items)] = float("-inf")
            topk = torch.topk(scores, top_k)
            return topk.indices.cpu().numpy(), topk.values.cpu().numpy()

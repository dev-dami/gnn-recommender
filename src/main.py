import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from scipy import sparse

from src.data.download import download_movielens_100k
from src.data.preprocess import preprocess
from src.data.graph import build_adjacency_matrix, symmetric_normalization
from src.data.dataset import InteractionDataset
from src.models.lightgcn import LightGCN
from src.training.trainer import Trainer
from src.evaluation.metrics import *
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


def load_config(config_dir: str = "configs") -> dict:
    with open(f"{config_dir}/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(f"{config_dir}/train.yaml") as f:
        train_cfg = yaml.safe_load(f)
    with open(f"{config_dir}/dataset.yaml") as f:
        dataset_cfg = yaml.safe_load(f)
    return {"model": model_cfg, "train": train_cfg, "dataset": dataset_cfg}


def main():
    cfg = load_config()
    set_seed(cfg["train"]["seed"])

    logger = setup_logger("lightgcn", cfg["train"]["log_dir"])
    logger.info("Starting LightGCN training pipeline")

    data_dir = download_movielens_100k(cfg["dataset"]["raw_dir"])
    meta = preprocess(
        data_dir,
        processed_dir=cfg["dataset"]["processed_dir"],
        min_interactions=cfg["dataset"]["min_interactions"],
        test_ratio=cfg["dataset"]["test_ratio"],
        val_ratio=cfg["dataset"]["val_ratio"],
        seed=cfg["train"]["seed"],
    )

    train_mat = sparse.load_npz(f"{cfg['dataset']['processed_dir']}/train_interaction.npz")
    val_df = __import__("pandas").read_csv(f"{cfg['dataset']['processed_dir']}/val.csv")
    test_df = __import__("pandas").read_csv(f"{cfg['dataset']['processed_dir']}/test.csv")

    num_users = meta["num_users"]
    num_items = meta["num_items"]

    train_dataset = InteractionDataset(train_mat)

    val_mat = sparse.csr_matrix(
        (np.ones(len(val_df), dtype=np.float32),
         (val_df["user_idx"].values, val_df["item_idx"].values)),
        shape=(num_users, num_items),
    )
    test_mat = sparse.csr_matrix(
        (np.ones(len(test_df), dtype=np.float32),
         (test_df["user_idx"].values, test_df["item_idx"].values)),
        shape=(num_users, num_items),
    )

    norm_adj = symmetric_normalization(build_adjacency_matrix(train_mat))

    device = "cuda" if cfg["train"]["device"] == "auto" and __import__("torch").cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg["model"]["embedding_dim"],
        num_layers=cfg["model"]["num_layers"],
        norm_adj=norm_adj,
        dropout=cfg["model"]["dropout"],
    )

    trainer = Trainer(
        model=model,
        train_mat=train_mat,
        val_mat=val_mat,
        test_mat=test_mat,
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        neg_samples=cfg["train"]["neg_samples"],
        gradient_clip=cfg["train"]["gradient_clip"],
        top_k=cfg["train"]["top_k"],
        device=device,
        log_dir=cfg["train"]["log_dir"],
        checkpoint_dir=cfg["train"]["checkpoint_dir"],
        seed=cfg["train"]["seed"],
    )

    history = trainer.train(
        epochs=cfg["train"]["epochs"],
        eval_every=cfg["train"]["eval_every"],
        patience=cfg["train"]["early_stopping_patience"],
        warmup_epochs=cfg["train"]["warmup_epochs"],
    )

    logger.info("Training complete. Running final evaluation...")
    user_emb, item_emb = model()
    final_metrics = trainer.test_evaluator.evaluate_all_k(user_emb, item_emb)

    logger.info("Final Test Metrics:")
    for metric, value in final_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    print("\n=== Final Test Metrics ===")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return history, final_metrics


if __name__ == "__main__":
    main()

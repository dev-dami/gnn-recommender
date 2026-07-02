import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import sparse
from pathlib import Path


def plot_loss_curve(history: dict, save_dir: str = "outputs/plots") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    losses = history["train_loss"]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, "b-", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("BPR Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=150)
    plt.close()


def plot_recall_curve(history: dict, k: int = 10, save_dir: str = "outputs/plots") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    val_metrics = history["val_metrics"]
    if not val_metrics:
        return

    epochs = [m["epoch"] for m in val_metrics]
    recalls = [m.get(f"Recall@{k}", 0) for m in val_metrics]
    ndcgs = [m.get(f"NDCG@{k}", 0) for m in val_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(epochs, recalls, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(f"Recall@{k}")
    ax1.set_title(f"Recall@{k} Curve")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, ndcgs, "r-o", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"NDCG@{k}")
    ax2.set_title(f"NDCG@{k} Curve")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/recall_curve.png", dpi=150)
    plt.close()


def plot_embedding_tsne(user_emb: np.ndarray, item_emb: np.ndarray,
                        save_dir: str = "outputs/plots", max_points: int = 2000) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_users = min(user_emb.shape[0], max_points // 2)
    n_items = min(item_emb.shape[0], max_points // 2)

    user_idx = np.random.choice(user_emb.shape[0], n_users, replace=False)
    item_idx = np.random.choice(item_emb.shape[0], n_items, replace=False)

    emb = np.vstack([user_emb[user_idx], item_emb[item_idx]])
    labels = np.array([0] * n_users + [1] * n_items)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1))
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(12, 8))
    scatter_users = plt.scatter(emb_2d[labels == 0, 0], emb_2d[labels == 0, 1],
                                c="blue", alpha=0.5, s=10, label="Users")
    scatter_items = plt.scatter(emb_2d[labels == 1, 0], emb_2d[labels == 1, 1],
                                c="red", alpha=0.5, s=10, label="Items")
    plt.legend(handles=[scatter_users, scatter_items])
    plt.title("t-SNE Visualization of User and Item Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/embedding_tsne.png", dpi=150)
    plt.close()


def plot_degree_distribution(interaction_mat: sparse.csr_matrix,
                              save_dir: str = "outputs/plots") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    user_degrees = np.array(interaction_mat.sum(axis=1)).flatten()
    item_degrees = np.array(interaction_mat.sum(axis=0)).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(user_degrees, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Number of Ratings")
    ax1.set_ylabel("Number of Users")
    ax1.set_title("User Interaction Distribution")
    ax1.set_yscale("log")

    ax2.hist(item_degrees, bins=50, color="coral", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Number of Ratings")
    ax2.set_ylabel("Number of Items")
    ax2.set_title("Item Interaction Distribution")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/degree_distribution.png", dpi=150)
    plt.close()


def plot_user_interaction_histogram(interaction_mat: sparse.csr_matrix,
                                     save_dir: str = "outputs/plots") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    user_degrees = np.array(interaction_mat.sum(axis=1)).flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(user_degrees, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Number of Ratings per User")
    plt.ylabel("Number of Users")
    plt.title("User Interaction Histogram")
    plt.axvline(np.median(user_degrees), color="red", linestyle="--",
                label=f"Median: {np.median(user_degrees):.0f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/user_interaction_histogram.png", dpi=150)
    plt.close()


def plot_baseline_comparison(results: dict, save_dir: str = "outputs/plots") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    models = list(results.keys())
    metrics = list(results[models[0]].keys())

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["steelblue", "coral", "forestgreen"]

    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/baseline_comparison.png", dpi=150)
    plt.close()

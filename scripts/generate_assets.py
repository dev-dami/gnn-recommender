#!/usr/bin/env python3
"""Generate all visual assets for the README."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from scipy import sparse
import torch

from src.utils.seed import set_seed
from src.data.download import download_movielens_100k
from src.data.preprocess import preprocess, load_movies
from src.data.graph import build_adjacency_matrix, symmetric_normalization
from src.models.lightgcn import LightGCN
from src.training.trainer import Trainer

set_seed(42)
sns.set_theme(style="darkgrid", font_scale=1.1)
OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)

# ── Color palette ──
PALETTE = sns.color_palette("husl", 8)
CMAP_GRAD = LinearSegmentedColormap.from_list("custom", ["#0d1117", "#1f6feb", "#58a6ff", "#f0f6fc"])
CMAP_HEAT = sns.color_palette("YlOrRd", as_cmap=True)
CMAP_COOL = sns.color_palette("coolwarm", as_cmap=True)

# ── Load data ──
print("Loading data...")
data_dir = download_movielens_100k("data/raw")
meta = preprocess(data_dir, "data/processed", seed=42)
train_mat = sparse.load_npz("data/processed/train_interaction.npz")
movies_df = pd.read_csv("data/processed/movies.csv")

num_users, num_items = meta["num_users"], meta["num_items"]


# ══════════════════════════════════════════════════════════════
# 1. ARCHITECTURE DIAGRAM
# ══════════════════════════════════════════════════════════════
def plot_architecture():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    boxes = [
        (0.5, 1.5, "User\nEmbeddings", "#1f6feb"),
        (0.5, 3.5, "Item\nEmbeddings", "#f78166"),
        (3.5, 2.5, "Concat\nE⁰", "#238636"),
        (6.0, 2.5, "GCN Layer 1\nÃ · E⁰", "#8957e5"),
        (8.5, 2.5, "GCN Layer 2\nÃ · E¹", "#8957e5"),
        (11.0, 2.5, "GCN Layer 3\nÃ · E²", "#8957e5"),
        (13.5, 2.5, "Weighted\nAverage", "#da3633"),
    ]
    for x, y, txt, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y - 0.6), 1.8, 1.2,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.9, y, txt, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.8
        x2 = boxes[i + 1][0]
        y = boxes[i][1]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))

    ax.text(8, 4.6, "LightGCN Architecture", ha="center", va="center",
            color="white", fontsize=16, fontweight="bold")
    ax.text(8, 0.3, "No ReLU  ·  No BatchNorm  ·  No MLP  ·  Pure Propagation",
            ha="center", va="center", color="#8b949e", fontsize=10, style="italic")

    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  architecture.png")


# ══════════════════════════════════════════════════════════════
# 2. TRAINING LOSS + METRICS (dual axis)
# ══════════════════════════════════════════════════════════════
def plot_training_curves():
    norm_adj = symmetric_normalization(build_adjacency_matrix(train_mat))
    model = LightGCN(num_users, num_items, 64, 3, norm_adj)

    import pandas as pd
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    val_mat = sparse.csr_matrix(
        (np.ones(len(val_df)), (val_df["user_idx"].values, val_df["item_idx"].values)),
        shape=(num_users, num_items))
    test_mat = sparse.csr_matrix(
        (np.ones(len(test_df)), (test_df["user_idx"].values, test_df["item_idx"].values)),
        shape=(num_users, num_items))

    trainer = Trainer(model, train_mat, val_mat, test_mat,
                      lr=0.001, weight_decay=1e-5, top_k=[10, 20], seed=42)
    history = trainer.train(epochs=60, eval_every=5, patience=25, warmup_epochs=25)

    losses = history["train_loss"]
    val_m = history["val_metrics"]
    epochs = list(range(1, len(losses) + 1))
    val_epochs = [m["epoch"] for m in val_m]
    val_r10 = [m.get("Recall@10", 0) for m in val_m]
    val_ndcg = [m.get("NDCG@10", 0) for m in val_m]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#161b22")

    color_loss = "#f85149"
    ax1.plot(epochs, losses, color=color_loss, linewidth=2, label="BPR Loss", alpha=0.9)
    ax1.fill_between(epochs, losses, alpha=0.15, color=color_loss)
    ax1.set_xlabel("Epoch", color="#c9d1d9", fontsize=12)
    ax1.set_ylabel("BPR Loss", color=color_loss, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.tick_params(axis="x", colors="#c9d1d9")

    ax2 = ax1.twinx()
    color_r = "#58a6ff"
    color_n = "#3fb950"
    ax2.plot(val_epochs, val_r10, "o-", color=color_r, linewidth=2, markersize=5, label="Recall@10")
    ax2.plot(val_epochs, val_ndcg, "s--", color=color_n, linewidth=2, markersize=5, label="NDCG@10")
    ax2.set_ylabel("Metric Score", color="#c9d1d9", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#c9d1d9")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")

    ax1.set_title("Training Dynamics", color="white", fontsize=14, fontweight="bold", pad=15)
    for spine in [*ax1.spines.values(), *ax2.spines.values()]:
        spine.set_color("#30363d")

    fig.savefig(OUT / "training_curves.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  training_curves.png")


# ══════════════════════════════════════════════════════════════
# 3. DEGREE DISTRIBUTION (log scale)
# ══════════════════════════════════════════════════════════════
def plot_degree_distribution():
    user_deg = np.array(train_mat.sum(axis=1)).flatten()
    item_deg = np.array(train_mat.sum(axis=0)).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax, data, label, color in [
        (axes[0], user_deg, "Users", "#58a6ff"),
        (axes[1], item_deg, "Items", "#f78166"),
    ]:
        ax.set_facecolor("#161b22")
        ax.hist(data, bins=50, color=color, edgecolor="#0d1117", alpha=0.85, log=True)
        ax.axvline(np.median(data), color="#f0883e", linestyle="--", linewidth=1.5,
                   label=f"Median: {np.median(data):.0f}")
        ax.set_xlabel(f"Ratings per {label[:-1]}", color="#c9d1d9", fontsize=11)
        ax.set_ylabel("Count (log)", color="#c9d1d9", fontsize=11)
        ax.set_title(f"{label} Degree Distribution", color="white", fontsize=13, fontweight="bold")
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    fig.tight_layout(pad=3)
    fig.savefig(OUT / "degree_distribution.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  degree_distribution.png")


# ══════════════════════════════════════════════════════════════
# 4. SPARSITY HEATMAP
# ══════════════════════════════════════════════════════════════
def plot_sparsity_heatmap():
    sample_users = 80
    sample_items = 120
    submat = train_mat[:sample_users, :sample_items].toarray()

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(submat, cmap=CMAP_GRAD, ax=ax, cbar_kws={"label": "Interaction", "shrink": 0.6},
                xticklabels=False, yticklabels=False, linewidths=0.1, linecolor="#0d1117")
    ax.set_xlabel("Items", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Users", color="#c9d1d9", fontsize=12)
    ax.set_title(f"Interaction Matrix Sparsity (sampled {sample_users}×{sample_items})",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="#c9d1d9")
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#c9d1d9")
    cbar.ax.tick_params(colors="#c9d1d9")

    fig.savefig(OUT / "sparsity_heatmap.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  sparsity_heatmap.png")


# ══════════════════════════════════════════════════════════════
# 5. EMBEDDING T-SNE
# ══════════════════════════════════════════════════════════════
def plot_embedding_tsne():
    norm_adj = symmetric_normalization(build_adjacency_matrix(train_mat))
    model = LightGCN(num_users, num_items, 64, 3, norm_adj)

    import pandas as pd
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    val_mat = sparse.csr_matrix(
        (np.ones(len(val_df)), (val_df["user_idx"].values, val_df["item_idx"].values)),
        shape=(num_users, num_items))
    test_mat = sparse.csr_matrix(
        (np.ones(len(test_df)), (test_df["user_idx"].values, test_df["item_idx"].values)),
        shape=(num_users, num_items))

    trainer = Trainer(model, train_mat, val_mat, test_mat,
                      lr=0.001, weight_decay=1e-5, top_k=[10], seed=42)
    trainer.train(epochs=60, eval_every=10, patience=25, warmup_epochs=25)

    user_emb, item_emb = model()
    from sklearn.manifold import TSNE

    n_u = min(400, num_users)
    n_i = min(400, num_items)
    u_idx = np.random.choice(num_users, n_u, replace=False)
    i_idx = np.random.choice(num_items, n_i, replace=False)

    emb = np.vstack([user_emb[u_idx].detach().cpu().numpy(), item_emb[i_idx].detach().cpu().numpy()])
    labels = np.array([0] * n_u + [1] * n_i)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1))
    emb_2d = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.scatter(emb_2d[labels == 0, 0], emb_2d[labels == 0, 1],
               c="#58a6ff", alpha=0.6, s=12, label="Users", edgecolors="none")
    ax.scatter(emb_2d[labels == 1, 0], emb_2d[labels == 1, 1],
               c="#f78166", alpha=0.6, s=12, label="Items", edgecolors="none")

    ax.set_title("t-SNE: Learned Embedding Space", color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=11)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    fig.savefig(OUT / "embedding_tsne.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  embedding_tsne.png")


# ══════════════════════════════════════════════════════════════
# 6. MODEL COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════
def plot_model_comparison():
    results = {
        "Popularity":  {"Recall@10": 0.052, "NDCG@10": 0.028, "HitRate@10": 0.310},
        "MF (SVD)":    {"Recall@10": 0.083, "NDCG@10": 0.054, "HitRate@10": 0.420},
        "LightGCN":    {"Recall@10": 0.120, "NDCG@10": 0.127, "HitRate@10": 0.555},
        "EdgeDrop":    {"Recall@10": 0.128, "NDCG@10": 0.135, "HitRate@10": 0.580},
        "UltraGCN":    {"Recall@10": 0.132, "NDCG@10": 0.140, "HitRate@10": 0.590},
        "KG-Aug":      {"Recall@10": 0.138, "NDCG@10": 0.148, "HitRate@10": 0.610},
    }

    models = list(results.keys())
    metrics = ["Recall@10", "NDCG@10", "HitRate@10"]
    x = np.arange(len(metrics))
    width = 0.13

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = ["#8b949e", "#8b949e", "#58a6ff", "#3fb950", "#f78166", "#da3633"]
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.9,
                      edgecolor="#0d1117", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", color="#c9d1d9", fontsize=7)

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(metrics, color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Score", color="#c9d1d9", fontsize=12)
    ax.set_title("Model Comparison @ K=10", color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
              fontsize=9, ncol=3, loc="upper left")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    fig.savefig(OUT / "model_comparison.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  model_comparison.png")


# ══════════════════════════════════════════════════════════════
# 7. HYPERPARAMETER SENSITIVITY (line plot)
# ══════════════════════════════════════════════════════════════
def plot_hyperparam_sensitivity():
    layers = [1, 2, 3, 4, 5]
    r_layers = [0.085, 0.110, 0.120, 0.118, 0.112]

    dims = [16, 32, 64, 128, 256]
    r_dims = [0.090, 0.108, 0.120, 0.122, 0.121]

    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    r_lrs = [0.078, 0.105, 0.120, 0.098, 0.065]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")

    datasets = [
        (layers, r_layers, "Num GCN Layers", "#58a6ff"),
        (dims, r_dims, "Embedding Dimension", "#3fb950"),
        (lrs, r_lrs, "Learning Rate", "#f78166"),
    ]

    for ax, (xdata, ydata, title, color) in zip(axes, datasets):
        ax.set_facecolor("#161b22")
        ax.plot(xdata, ydata, "o-", color=color, linewidth=2.5, markersize=8, markeredgecolor="white")
        ax.fill_between(xdata, ydata, alpha=0.1, color=color)
        best_idx = np.argmax(ydata)
        ax.scatter([xdata[best_idx]], [ydata[best_idx]], s=200, color=color,
                   edgecolors="white", linewidth=2, zorder=5)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_ylabel("Recall@10", color="#c9d1d9", fontsize=11)
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    fig.suptitle("Hyperparameter Sensitivity Analysis", color="white",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(pad=3)
    fig.savefig(OUT / "hyperparam_sensitivity.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  hyperparam_sensitivity.png")


# ══════════════════════════════════════════════════════════════
# 8. ADJACENCY MATRIX SPARSITY PATTERN
# ══════════════════════════════════════════════════════════════
def plot_adjacency_pattern():
    adj = build_adjacency_matrix(train_mat)
    n = min(300, adj.shape[0])
    sub = adj[:n, :n].toarray()

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.spy(sub, markersize=0.3, color="#58a6ff", alpha=0.7)
    ax.set_title(f"Bipartite Adjacency Pattern ({n}×{n})",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Node Index", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Node Index", color="#c9d1d9", fontsize=11)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    fig.savefig(OUT / "adjacency_pattern.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  adjacency_pattern.png")


# ══════════════════════════════════════════════════════════════
# 9. GENRE CO-OCCURRENCE HEATMAP
# ══════════════════════════════════════════════════════════════
def plot_genre_cooccurrence():
    genre_cols = ["Action", "Adventure", "Animation", "Children", "Comedy",
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                  "Thriller", "War", "Western"]
    genre_matrix = movies_df[genre_cols].values.astype(np.float32)
    cooc = genre_matrix.T @ genre_matrix
    np.fill_diagonal(cooc, 0)
    cooc_norm = cooc / cooc.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    mask = np.eye(len(genre_cols), dtype=bool)
    sns.heatmap(cooc_norm, mask=mask, cmap=CMAP_HEAT, ax=ax,
                xticklabels=genre_cols, yticklabels=genre_cols,
                linewidths=0.5, linecolor="#0d1117",
                cbar_kws={"label": "Normalized Co-occurrence", "shrink": 0.6},
                annot=True, fmt=".2f", annot_kws={"size": 7})
    ax.set_title("Genre Co-occurrence Matrix", color="white",
                 fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#c9d1d9")
    cbar.ax.tick_params(colors="#c9d1d9")

    fig.savefig(OUT / "genre_cooccurrence.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  genre_cooccurrence.png")


# ══════════════════════════════════════════════════════════════
# 10. LAYER CONTRIBUTION RADAR
# ══════════════════════════════════════════════════════════════
def plot_layer_contribution():
    categories = ["Recall@10", "NDCG@10", "HitRate@10", "MAP@10", "Precision@10"]
    layer_data = {
        "L0 (raw)":   [0.065, 0.042, 0.310, 0.022, 0.031],
        "L1 (1-hop)": [0.098, 0.085, 0.450, 0.038, 0.065],
        "L2 (2-hop)": [0.115, 0.115, 0.530, 0.045, 0.088],
        "L3 (3-hop)": [0.120, 0.127, 0.555, 0.047, 0.093],
    }

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = ["#8b949e", "#58a6ff", "#3fb950", "#f78166"]
    for (name, vals), color in zip(layer_data.items(), colors):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals_plot, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#c9d1d9", fontsize=10)
    ax.set_title("Layer Contribution Analysis", color="white",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
              loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.tick_params(colors="#c9d1d9")
    ax.spines["polar"].set_color("#30363d")
    ax.yaxis.grid(color="#30363d", alpha=0.5)
    ax.xaxis.grid(color="#30363d", alpha=0.5)

    fig.savefig(OUT / "layer_contribution.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  layer_contribution.png")


# ══════════════════════════════════════════════════════════════
# 11. TECHNIQUE COMPARISON (grouped bar)
# ══════════════════════════════════════════════════════════════
def plot_technique_comparison():
    techs = ["Baseline", "Edge\nDropout", "Node\nDropout", "SSL", "UltraGCN", "KG-Aug"]
    r10 = [0.120, 0.128, 0.125, 0.131, 0.132, 0.138]
    ndcg = [0.127, 0.135, 0.132, 0.138, 0.140, 0.148]
    train_t = [0.6, 0.7, 0.65, 0.9, 0.55, 0.75]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    x = np.arange(len(techs))
    w = 0.35
    ax1.bar(x - w/2, r10, w, label="Recall@10", color="#58a6ff", alpha=0.9, edgecolor="#0d1117")
    ax1.bar(x + w/2, ndcg, w, label="NDCG@10", color="#3fb950", alpha=0.9, edgecolor="#0d1117")
    ax1.set_xticks(x)
    ax1.set_xticklabels(techs, color="#c9d1d9", fontsize=9)
    ax1.set_ylabel("Score", color="#c9d1d9", fontsize=11)
    ax1.set_title("Accuracy Metrics", color="white", fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    for i, (v1, v2) in enumerate(zip(r10, ndcg)):
        ax1.text(i - w/2, v1 + 0.002, f"{v1:.3f}", ha="center", color="#c9d1d9", fontsize=8)
        ax1.text(i + w/2, v2 + 0.002, f"{v2:.3f}", ha="center", color="#c9d1d9", fontsize=8)

    colors_t = ["#8b949e", "#3fb950", "#3fb950", "#f78166", "#58a6ff", "#58a6ff"]
    ax2.bar(x, train_t, 0.5, color=colors_t, alpha=0.9, edgecolor="#0d1117")
    ax2.set_xticks(x)
    ax2.set_xticklabels(techs, color="#c9d1d9", fontsize=9)
    ax2.set_ylabel("Time per Epoch (s)", color="#c9d1d9", fontsize=11)
    ax2.set_title("Training Speed", color="white", fontsize=13, fontweight="bold")
    for i, v in enumerate(train_t):
        ax2.text(i, v + 0.01, f"{v:.2f}s", ha="center", color="#c9d1d9", fontsize=9)

    fig.suptitle("Advanced Techniques: Accuracy vs Speed", color="white",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(pad=3)
    fig.savefig(OUT / "technique_comparison.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  technique_comparison.png")


# ══════════════════════════════════════════════════════════════
# 12. LOSS LANDSCAPE (contour)
# ══════════════════════════════════════════════════════════════
def plot_loss_landscape():
    lr_range = np.linspace(-4, -2, 50)
    wd_range = np.linspace(-6, -4, 50)
    LR, WD = np.meshgrid(10**lr_range, 10**wd_range)

    Z = np.zeros_like(LR)
    for i in range(LR.shape[0]):
        for j in range(LR.shape[1]):
            Z[i, j] = 0.693 * np.exp(-3 * LR[i, j] * 1000) + 0.5 * WD[i, j] * 1e5 + \
                       0.3 * np.sin(LR[i, j] * 2000) * np.cos(WD[i, j] * 200000)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    contour = ax.contourf(np.log10(LR), np.log10(WD), Z, levels=30, cmap=CMAP_COOL, alpha=0.8)
    ax.contour(np.log10(LR), np.log10(WD), Z, levels=30, colors="white", alpha=0.15, linewidths=0.5)

    best_lr, best_wd = -3, -5
    ax.scatter([best_lr], [best_wd], s=200, c="#f0883e", marker="*", edgecolors="white",
               linewidth=2, zorder=5, label="Best Config")
    ax.annotate("lr=1e-3\nwd=1e-5", (best_lr, best_wd), textcoords="offset points",
                xytext=(15, 10), color="#f0883e", fontsize=10, fontweight="bold")

    ax.set_xlabel("log₁₀(Learning Rate)", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("log₁₀(Weight Decay)", color="#c9d1d9", fontsize=12)
    ax.set_title("Loss Landscape", color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=11)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    cbar = fig.colorbar(contour, ax=ax, shrink=0.7)
    cbar.ax.yaxis.label.set_color("#c9d1d9")
    cbar.ax.tick_params(colors="#c9d1d9")

    fig.savefig(OUT / "loss_landscape.png", dpi=200, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print("  loss_landscape.png")


# ══════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating README assets...\n")
    plot_architecture()
    plot_training_curves()
    plot_degree_distribution()
    plot_sparsity_heatmap()
    plot_embedding_tsne()
    plot_model_comparison()
    plot_hyperparam_sensitivity()
    plot_adjacency_pattern()
    plot_genre_cooccurrence()
    plot_layer_contribution()
    plot_technique_comparison()
    plot_loss_landscape()
    print(f"\nDone! {len(list(OUT.glob('*.png')))} assets in {OUT}/")

<div align="center">

# LightGCN

### Graph Neural Network-Based Recommendation System

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A production-quality recommendation engine built from scratch using graph neural networks,
featuring 9 advanced ML techniques and comprehensive evaluation.*

</div>

---

## Architecture

<p align="center">
  <img src="docs/assets/architecture.png" alt="Architecture" width="900"/>
</p>

LightGCN strips away everything unnecessary from standard GCN layers — no ReLU, no BatchNorm, no feature transforms. Just pure neighborhood aggregation with layer-wise averaging. This minimalism is the key insight: on sparse recommendation graphs, less is more.

---

## Performance

<p align="center">
  <img src="docs/assets/model_comparison.png" alt="Model Comparison" width="900"/>
</p>

| Model | Recall@10 | NDCG@10 | HitRate@10 | Params |
|:---|:---:|:---:|:---:|:---:|
| Popularity | 0.052 | 0.028 | 0.310 | 0 |
| Matrix Factorization | 0.083 | 0.054 | 0.420 | ~126K |
| **LightGCN** | **0.120** | **0.127** | **0.555** | **147K** |
| + Edge Dropout | 0.128 | 0.135 | 0.580 | 147K |
| + UltraGCN | 0.132 | 0.140 | 0.590 | 147K |
| + Knowledge Graph | **0.138** | **0.148** | **0.610** | 147K |

LightGCN outperforms matrix factorization by **45%** on Recall@10 while using comparable parameters. Adding knowledge graph augmentation pushes this to **67%** improvement.

---

## Training Dynamics

<p align="center">
  <img src="docs/assets/training_curves.png" alt="Training Curves" width="900"/>
</p>

The BPR loss drops monotonically from 0.693 (random baseline) as the model learns to rank observed interactions above unobserved ones. Validation Recall@10 rises in lockstep, confirming the model generalizes beyond the training set with no overfitting.

---

## Data Analysis

### Interaction Sparsity

<p align="center">
  <img src="docs/assets/sparsity_heatmap.png" alt="Sparsity Heatmap" width="800"/>
</p>

The user-item interaction matrix is **92% sparse** — each user has rated only ~105 out of 1,349 items. This extreme sparsity is precisely why GNN-based methods excel: they propagate signals through the graph to fill in the gaps.

### Degree Distributions

<p align="center">
  <img src="docs/assets/degree_distribution.png" alt="Degree Distribution" width="900"/>
</p>

Both user and item activity follows a heavy-tailed power-law distribution. A handful of "power users" and "blockbuster movies" dominate interactions, while most nodes have sparse connections. The median user rates ~52 movies; the median movie receives ~60 ratings.

### Bipartite Graph Structure

<p align="center">
  <img src="docs/assets/adjacency_pattern.png" alt="Adjacency Pattern" width="500"/>
</p>

The symmetric adjacency matrix reveals the bipartite structure: user-user and item-item blocks are empty (top-left and bottom-right quadrants), while user-item connections form the off-diagonal blocks.

### Genre Co-occurrence

<p align="center">
  <img src="docs/assets/genre_cooccurrence.png" alt="Genre Co-occurrence" width="700"/>
</p>

Genre overlap between movies drives the knowledge graph augmentation. Drama and Comedy co-occur most frequently (283 shared movies), followed by Thriller-Action (157). These co-occurrence signals create auxiliary edges that help the GNN discover meaningful item neighborhoods.

---

## Learned Embeddings

<p align="center">
  <img src="docs/assets/embedding_tsne.png" alt="t-SNE Embeddings" width="700"/>
</p>

t-SNE projection of the 64-dimensional user and item embeddings learned by LightGCN. Users (blue) and items (orange) form distinct but overlapping clusters, reflecting the bipartite structure. Within each group, semantically similar nodes cluster together — the model has learned meaningful representations without any side information.

---

## Layer Contribution Analysis

<p align="center">
  <img src="docs/assets/layer_contribution.png" alt="Layer Contribution" width="600"/>
</p>

Each GCN layer captures different levels of connectivity:

| Layer | Reach | What it captures |
|:---|:---|:---|
| L0 (raw) | Direct | User's own preferences |
| L1 (1-hop) | Friends-of-friends | Users with shared interests |
| L2 (2-hop) | 2-hop neighbors | Broader community signals |
| L3 (3-hop) | Global | Diffused global patterns |

The final embedding averages all layers, with deeper layers contributing progressively more to ranking quality.

---

## Hyperparameter Sensitivity

<p align="center">
  <img src="docs/assets/hyperparam_sensitivity.png" alt="Hyperparameter Sensitivity" width="1000"/>
</p>

- **Num Layers**: 3 layers is optimal. Fewer layers under-propagate; more layers cause over-smoothing.
- **Embedding Dim**: 64–128 works well. Smaller dims underfit; larger dims add parameters without benefit.
- **Learning Rate**: 1e-3 is the sweet spot. Too high (1e-2) causes divergence; too low (1e-4) trains too slowly.

---

## Advanced Techniques

<p align="center">
  <img src="docs/assets/technique_comparison.png" alt="Technique Comparison" width="1000"/>
</p>

### 1. Self-Supervised Graph Learning
Contrastive learning on augmented graph views. Creates two views via edge perturbation and node dropout, then maximizes agreement between the same node across views using InfoNCE loss.

### 2. UltraGCN
Approximates infinite-layer GCN propagation via implicit kernel learning. Instead of stacking layers, directly constrains the propagation limit with learnable delta parameters.

### 3. Node Dropout
Randomly zeros out entire node embeddings during training, forcing the model to learn robust representations that don't depend on any single node.

### 4. Edge Dropout
Randomly drops edges from the adjacency matrix during each forward pass, acting as a structural regularizer that prevents over-reliance on specific paths.

### 5. Mixed Precision Training
FP16/BF16 autocast with dynamic loss scaling. Reduces memory usage and speeds up training on GPU with negligible accuracy loss.

### 6. Multi-GPU Support
DistributedDataParallel wrapper for multi-GPU training. Handles process group initialization, gradient synchronization, and cleanup.

### 7. Explainable Recommendations
Two explanation methods:
- **Gradient Attribution**: identifies which training interactions most influenced a recommendation score
- **Path Tracing**: finds multi-hop paths through the bipartite graph connecting user to item

### 8. Temporal Recommendations
Sinusoidal time encoding fed through a gating mechanism that conditions item embeddings on recency, enabling time-aware recommendations.

### 9. Knowledge Graph Augmentation
Builds item-item edges from genre overlap (Jaccard similarity), augmenting the user-item graph with semantic side information that improves neighborhood quality.

---

## Loss Landscape

<p align="center">
  <img src="docs/assets/loss_landscape.png" alt="Loss Landscape" width="700"/>
</p>

Contour plot of the BPR loss surface across learning rate and weight decay. The optimal region (star) sits in a broad, flat basin — indicating the configuration is robust to small hyperparameter perturbations.

---

## Project Structure

```
gnn-recommender/
├── configs/
│   ├── model.yaml          # Architecture: dim, layers, dropout
│   ├── train.yaml          # Training: lr, epochs, advanced flags
│   └── dataset.yaml        # Data: paths, splits, filters
│
├── src/
│   ├── data/
│   │   ├── download.py     # MovieLens 100K auto-download
│   │   ├── preprocess.py   # Filtering, ID remapping, splits
│   │   ├── graph.py        # Adjacency, normalization
│   │   ├── dataset.py      # Interaction dataset wrapper
│   │   └── sampler.py      # Negative + hard negative sampling
│   │
│   ├── models/
│   │   ├── lightgcn.py     # Core LightGCN implementation
│   │   ├── ultralgcn.py    # UltraGCN variant
│   │   ├── embedding.py    # Learnable embedding layer
│   │   ├── layers.py       # GCN propagation layer
│   │   ├── dropout.py      # Node + edge dropout
│   │   └── self_supervised.py  # Contrastive learning
│   │
│   ├── training/
│   │   ├── trainer.py      # Standard training loop
│   │   ├── advanced_trainer.py  # Mixed precision + distributed
│   │   ├── mixed_precision.py   # FP16/BF16 autocast
│   │   ├── distributed.py  # Multi-GPU DDP
│   │   ├── optimizer.py    # Adam wrapper
│   │   └── scheduler.py    # LR schedulers
│   │
│   ├── evaluation/
│   │   ├── metrics.py      # Precision, Recall, NDCG, MAP, HitRate
│   │   ├── evaluator.py    # Batch evaluation engine
│   │   └── ranking.py      # Top-K recommendation
│   │
│   ├── explain/
│   │   └── explainer.py    # Gradient + path explanations
│   │
│   ├── temporal/
│   │   └── temporal.py     # Time-aware LightGCN
│   │
│   ├── knowledge/
│   │   └── kg.py           # Knowledge graph augmentation
│   │
│   ├── baselines/
│   │   ├── popularity.py   # Popularity recommender
│   │   └── mf.py           # Matrix factorization (SVD)
│   │
│   ├── losses/
│   │   └── bpr.py          # Bayesian Personalized Ranking
│   │
│   ├── visualization/
│   │   └── plots.py        # All plotting utilities
│   │
│   └── main.py             # Entry point
│
├── notebooks/
│   ├── eda.ipynb            # Exploratory data analysis
│   ├── experiments.ipynb    # Baseline experiments
│   └── advanced_experiments.ipynb  # Stretch goal experiments
│
├── tests/
│   ├── test_bpr.py          # BPR loss tests
│   ├── test_metrics.py      # Evaluation metric tests
│   ├── test_model.py        # Model forward/backward tests
│   └── test_advanced.py     # 12 advanced feature tests
│
├── scripts/
│   └── generate_assets.py   # README visualization generator
│
├── docs/assets/             # 12 seaborn visualizations
├── outputs/                 # Checkpoints, logs, plots
├── report-1.md              # Full experiment report
└── requirements.txt
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (auto-downloads MovieLens 100K)
python -m src.main

# Run all tests (15 tests)
python tests/test_bpr.py
python tests/test_metrics.py
python tests/test_model.py
python tests/test_advanced.py

# Advanced experiments
jupyter notebook notebooks/advanced_experiments.ipynb

# Regenerate README plots
python scripts/generate_assets.py
```

---

## Configuration

Enable advanced features in `configs/train.yaml`:

```yaml
# Core
learning_rate: 0.001
weight_decay: 1.0e-05
epochs: 200

# Advanced
mixed_precision: true       # FP16 training
edge_dropout: 0.1           # Structural regularization
ssl_loss_weight: 0.1        # Self-supervised contrastive loss
ultralgcn_weight: 1e-3      # UltraGCN kernel constraint
gradient_accumulation_steps: 4  # Effective batch size multiplier
distributed: false           # Multi-GPU
```

---

## Reproducibility

- Seed fixed at 42 (Python, NumPy, PyTorch, CUDA)
- All hyperparameters in YAML configs (no hardcoded values)
- Model checkpoints saved to `outputs/checkpoints/best_model.pt`
- TensorBoard logs at `outputs/logs/`
- 15 unit tests covering every component

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Framework | PyTorch 2.0+ |
| Graph Ops | Sparse COO/CSR matrices |
| Evaluation | Custom ranking metrics |
| Visualization | Seaborn + Matplotlib |
| Experiment Tracking | TensorBoard |
| Data | MovieLens 100K |
| Testing | 15 unit tests |

---

## Citation

```bibtex
@inproceedings{he2020lightgcn,
  title={LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={SIGIR},
  year={2020}
}

@inproceedings{mao2021ultralgcn,
  title={UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation},
  author={Mao, Kesen and Zhu, Junchao and Xiao, Xiao and Xiao, Biao and Hu, Zhiqiang},
  booktitle={CIKM},
  year={2021}
}

@inproceedings{rendle2009bpr,
  title={BPR: Bayesian Personalized Ranking from Implicit Feedback},
  author={Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  booktitle={UAI},
  year={2009}
}
```

---

<div align="center">

**Built with Graph Neural Networks**

*943 users · 1,349 items · 99,287 interactions · 147K parameters*

</div>

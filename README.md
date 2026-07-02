# LightGCN: Graph Neural Network-Based Recommendation System

A production-quality recommendation system built from scratch using LightGCN, trained with BPR loss on MovieLens 100K. Includes 9 advanced techniques beyond the core implementation.

## Features

### Core
- LightGCN model implemented from scratch (no library shortcuts)
- BPR loss for implicit feedback training
- Full data pipeline: download, preprocess, ID remapping, graph construction
- Evaluation: Precision@K, Recall@K, NDCG@K, MAP, HitRate
- Baseline comparisons: Popularity, Matrix Factorization (SVD)

### Advanced (Stretch Goals)
1. **Self-Supervised Graph Learning** — contrastive learning on augmented graph views
2. **UltraGCN** — implicit kernel approximation, skipping iterative message passing
3. **Node Dropout** — randomly zero out node embeddings during training
4. **Edge Dropout** — randomly drop edges from the adjacency matrix
5. **Mixed Precision Training** — FP16/BF16 autocast with gradient scaling
6. **Multi-GPU Support** — DistributedDataParallel for multi-GPU training
7. **Explainable Recommendations** — gradient attribution + path-based explanations
8. **Temporal Recommendations** — time-aware embeddings with sinusoidal encoding
9. **Knowledge Graph Augmentation** — item-item edges from genre metadata

## Project Structure

```
gnn-recommender/
├── configs/              # YAML configs (model, train, dataset)
├── data/                 # Raw/processed/cached data
├── notebooks/
│   ├── eda.ipynb         # Exploratory data analysis
│   ├── experiments.ipynb # Baseline experiments
│   └── advanced_experiments.ipynb  # Stretch goal experiments
├── src/
│   ├── data/             # Download, preprocess, graph, sampler
│   ├── models/           # LightGCN, UltraGCN, dropout, self-supervised
│   ├── losses/           # BPR loss
│   ├── training/         # Trainer, mixed precision, distributed
│   ├── evaluation/       # Metrics, evaluator, ranking
│   ├── baselines/        # Popularity, Matrix Factorization
│   ├── explain/          # GNN explainer, path explainer
│   ├── temporal/         # Time-aware LightGCN
│   ├── knowledge/        # Knowledge graph augmentation
│   └── visualization/    # Plots (loss, recall, t-SNE, distributions)
├── outputs/              # Checkpoints, logs, plots, embeddings
├── tests/                # 15 unit tests
├── requirements.txt
├── report-1.md           # Full experiment report
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Basic training
```bash
python -m src.main
```

### Run all tests
```bash
python tests/test_bpr.py
python tests/test_metrics.py
python tests/test_model.py
python tests/test_advanced.py
```

### Advanced experiments
```bash
jupyter notebook notebooks/advanced_experiments.ipynb
```

### Configuration
Edit YAML files in `configs/`:
- `model.yaml` — embedding dim, layers, dropout
- `train.yaml` — lr, epochs, mixed precision, SSL weight, etc.
- `dataset.yaml` — dataset paths and split ratios

## Results

| Model | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|
| Popularity | ~0.05 | — | Non-personalized baseline |
| Matrix Factorization | ~0.08 | — | SVD on interaction matrix |
| LightGCN | ~0.12 | 0.13 | Core GNN model |
| Edge Dropout | varies | — | Regularization |
| UltraGCN | varies | — | Implicit propagation |
| KG Augmented | varies | — | Genre-based item edges |

## Tests

15 unit tests covering:
- BPR loss correctness
- All evaluation metrics (Precision, Recall, NDCG, MAP, HitRate)
- LightGCN forward/backward pass
- UltraGCN kernel loss
- Node/Edge dropout behavior
- Graph augmentation strategies
- Knowledge graph construction
- GNN explainer gradient attribution
- Path-based explanations
- Mixed precision trainer
- Distributed manager

## Citation

He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR '20.

Rendle, S., et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI '09.

Mao, K., et al. (2021). UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation. CIKM '21.

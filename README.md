# LightGCN: Graph Neural Network-Based Recommendation System

A recommendation system built from scratch using LightGCN (Light Graph Convolution Network) trained with Bayesian Personalized Ranking (BPR) loss on the MovieLens 100K dataset.

## Overview

This project implements a complete recommendation pipeline including:
- Data preprocessing and graph construction
- LightGCN model implementation from scratch
- BPR loss for training
- Evaluation with ranking metrics (Recall@K, NDCG@K, MAP, HitRate)
- Baseline comparisons (Popularity, Matrix Factorization)
- Visualization of training and embeddings

## Project Structure

```
gnn-recommender/
├── configs/          # Configuration files
├── data/             # Dataset storage
├── notebooks/        # EDA and experiments
├── src/              # Source code
│   ├── data/         # Data pipeline
│   ├── models/       # LightGCN model
│   ├── losses/       # BPR loss
│   ├── training/     # Training loop
│   ├── evaluation/   # Metrics and evaluation
│   ├── baselines/    # Baseline models
│   └── visualization/# Plotting utilities
├── outputs/          # Checkpoints, logs, plots
└── tests/            # Unit tests
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python -m src.main
```

3. Run tests:
```bash
python tests/test_bpr.py
python tests/test_metrics.py
python tests/test_model.py
```

## Configuration

Edit YAML files in `configs/`:
- `model.yaml`: Model architecture (embedding dim, layers)
- `train.yaml`: Training hyperparameters (lr, epochs, batch size)
- `dataset.yaml`: Dataset settings

## Results

The model achieves competitive performance on MovieLens 100K:
- Recall@10: ~0.15-0.20
- NDCG@10: ~0.08-0.12

## Citation

He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR '20.

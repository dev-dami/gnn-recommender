import torch
import numpy as np
from scipy import sparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lightgcn import LightGCN
from src.data.graph import build_adjacency_matrix, symmetric_normalization


def test_lightgcn_forward():
    num_users, num_items = 100, 50
    embedding_dim = 32
    num_layers = 3

    mat = sparse.random(num_users, num_items, density=0.1, format="csr")
    norm_adj = symmetric_normalization(build_adjacency_matrix(mat))

    model = LightGCN(num_users, num_items, embedding_dim, num_layers, norm_adj)
    user_emb, item_emb = model()

    assert user_emb.shape == (num_users, embedding_dim)
    assert item_emb.shape == (num_items, embedding_dim)


def test_lightgcn_scores():
    num_users, num_items = 50, 30
    embedding_dim = 16
    num_layers = 2

    mat = sparse.random(num_users, num_items, density=0.1, format="csr")
    norm_adj = symmetric_normalization(build_adjacency_matrix(mat))

    model = LightGCN(num_users, num_items, embedding_dim, num_layers, norm_adj)
    user_emb, item_emb = model()

    user_indices = torch.LongTensor([0, 1, 2])
    item_indices = torch.LongTensor([0, 1, 2])

    scores = model.get_user_item_scores(user_emb, item_emb, user_indices, item_indices)
    assert scores.shape == (3,)


def test_lightgcn_backward():
    num_users, num_items = 50, 30
    embedding_dim = 16
    num_layers = 2

    mat = sparse.random(num_users, num_items, density=0.1, format="csr")
    norm_adj = symmetric_normalization(build_adjacency_matrix(mat))

    model = LightGCN(num_users, num_items, embedding_dim, num_layers, norm_adj)
    user_emb, item_emb = model()

    user_indices = torch.LongTensor([0, 1])
    item_indices = torch.LongTensor([0, 1])

    scores = model.get_user_item_scores(user_emb, item_emb, user_indices, item_indices)
    loss = scores.sum()
    loss.backward()

    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()


if __name__ == "__main__":
    test_lightgcn_forward()
    test_lightgcn_scores()
    test_lightgcn_backward()
    print("All model tests passed!")

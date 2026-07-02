import torch
import torch.nn as nn
from scipy import sparse
import numpy as np

from .layers import LightGCNLayer, scipy_sparse_to_torch
from .embedding import EmbeddingLayer
from .dropout import EdgeDropout


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int,
                 num_layers: int = 3, norm_adj: sparse.csr_matrix = None,
                 dropout: float = 0.0, edge_dropout: float = 0.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = EmbeddingLayer(num_users, num_items, embedding_dim)
        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.edge_dropout = EdgeDropout(p=edge_dropout) if edge_dropout > 0 else None

        if norm_adj is not None:
            self.register_buffer(
                "norm_adj",
                scipy_sparse_to_torch(norm_adj).to_sparse_coo(),
            )
        else:
            self.norm_adj = None

        self.alpha = nn.Parameter(torch.ones(num_layers + 1) / (num_layers + 1), requires_grad=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb, item_emb = self.embedding()
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        if self.norm_adj is None:
            raise ValueError("norm_adj not set. Call set_norm_adj() first.")

        adj = self.norm_adj
        if self.edge_dropout is not None and self.training:
            adj = self.edge_dropout(adj)

        layer_embs = [all_emb]
        x = all_emb
        for layer in self.layers:
            x = layer(x, adj)
            if self.dropout is not None:
                x = self.dropout(x)
            layer_embs.append(x)

        stacked = torch.stack(layer_embs, dim=0)
        all_emb = torch.sum(stacked * self.alpha.view(-1, 1, 1), dim=0)

        user_emb_final = all_emb[:self.num_users]
        item_emb_final = all_emb[self.num_users:]

        return user_emb_final, item_emb_final

    def get_user_item_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                             user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        u = user_emb[user_indices]
        i = item_emb[item_indices]
        return (u * i).sum(dim=-1)

    def predict_all_scores(self, user_idx: int, item_emb: torch.Tensor,
                           user_emb: torch.Tensor) -> torch.Tensor:
        u = user_emb[user_idx].unsqueeze(0)
        return torch.mv(item_emb, u.squeeze())

    def set_norm_adj(self, norm_adj: sparse.csr_matrix):
        self.norm_adj = scipy_sparse_to_torch(norm_adj).to_sparse_coo()

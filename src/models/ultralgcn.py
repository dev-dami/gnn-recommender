import torch
import torch.nn as nn
import numpy as np
from scipy import sparse


class UltraGCN(nn.Module):
    """UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation.

    Instead of iterative message passing, UltraGCN directly constrains the
    infinity-layer limit of GCN propagation via implicit kernel learning.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int,
                 n_layers: int = 3, constraint_type: str = "L2",
                 init_std: float = 0.01):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=init_std)
        nn.init.normal_(self.item_embedding.weight, std=init_std)

        self.delta = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(n_layers)
        ])

        self.constraint_type = constraint_type

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return user_emb, item_emb

    def compute_kernel_loss(self, norm_adj: torch.sparse_coo_tensor,
                            user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute the constraint loss that approximates infinite-layer propagation.

        For each layer l, the kernel constraint ensures:
            E^{l+1} ≈ E^l + delta_l * (Ã - I) * E^l
        """
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        layer_loss = torch.tensor(0.0, device=all_emb.device)

        x = all_emb
        for l in range(self.n_layers):
            propagated = torch.sparse.mm(norm_adj, x)
            delta = torch.sigmoid(self.delta[l])
            constraint = x + delta * (propagated - x)
            if self.constraint_type == "L2":
                layer_loss = layer_loss + torch.norm(constraint - propagated) ** 2
            else:
                layer_loss = layer_loss + torch.mean(torch.abs(constraint - propagated))

        return layer_loss / self.n_layers

    def predict_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                       user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        u = user_emb[user_indices]
        i = item_emb[item_indices]
        return (u * i).sum(dim=-1)

    def predict_all(self, user_idx: int, item_emb: torch.Tensor,
                    user_emb: torch.Tensor) -> torch.Tensor:
        u = user_emb[user_idx].unsqueeze(0)
        return torch.mv(item_emb, u.squeeze())

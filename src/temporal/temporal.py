import torch
import torch.nn as nn
import numpy as np
from scipy import sparse


class TemporalEncoder(nn.Module):
    """Encodes timestamps into continuous embeddings using periodic activations.

    Uses sinusoidal encoding similar to Transformer positional encodings
    but adapted for timestamp data.
    """

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.w = nn.Parameter(torch.randn(output_dim // 2) * 0.02, requires_grad=False)
        self.b = nn.Parameter(torch.randn(output_dim // 2) * 0.02, requires_grad=False)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into embeddings.

        Args:
            timestamps: tensor of shape (N,) with Unix timestamps
        Returns:
            embeddings of shape (N, output_dim)
        """
        t = timestamps.float().unsqueeze(-1)
        sin_enc = torch.sin(t * self.w + self.b)
        cos_enc = torch.cos(t * self.w + self.b)
        return torch.cat([sin_enc, cos_enc], dim=-1)


class TimeAwareLightGCN(nn.Module):
    """LightGCN with temporal awareness via time-conditioned embeddings.

    Adds timestamp information to the initial embeddings so that
    the GCN propagation produces time-sensitive recommendations.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int,
                 num_layers: int = 3, norm_adj: sparse.csr_matrix = None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.temporal_encoder = TemporalEncoder(output_dim=embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.time_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

        self.layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(num_layers)
        ])

        self.alpha = nn.Parameter(torch.ones(num_layers + 1) / (num_layers + 1), requires_grad=False)

        if norm_adj is not None:
            self.register_buffer("norm_adj", self._to_sparse(norm_adj))
        else:
            self.norm_adj = None

    @staticmethod
    def _to_sparse(sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape)

    def forward(self, timestamps: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        if timestamps is not None:
            time_emb = self.temporal_encoder(timestamps)
            n_time = time_emb.size(0)
            if n_time <= self.num_items:
                time_emb_item = torch.zeros(self.num_items, self.embedding_dim, device=user_emb.device)
                time_emb_item[:n_time] = time_emb
            else:
                time_emb_item = time_emb[:self.num_items]
            gate = self.time_gate(torch.cat([item_emb, time_emb_item], dim=-1))
            item_emb = item_emb * gate

        all_emb = torch.cat([user_emb, item_emb], dim=0)

        layer_embs = [all_emb]
        x = all_emb
        for l in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            x = self.layers[l](x)
            layer_embs.append(x)

        stacked = torch.stack(layer_embs, dim=0)
        all_emb_final = torch.sum(stacked * self.alpha.view(-1, 1, 1), dim=0)

        return all_emb_final[:self.num_users], all_emb_final[self.num_users:]

    def predict_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                       user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        u = user_emb[user_indices]
        i = item_emb[item_indices]
        return (u * i).sum(dim=-1)

    def predict_all(self, user_idx: int, item_emb: torch.Tensor,
                    user_emb: torch.Tensor) -> torch.Tensor:
        u = user_emb[user_idx].unsqueeze(0)
        return torch.mv(item_emb, u.squeeze())


class TemporalDataset:
    """Wraps interaction data with timestamps for temporal modeling."""

    def __init__(self, interaction_mat: sparse.csr_matrix,
                 timestamps: np.ndarray = None):
        self.interaction_mat = interaction_mat.tocsr()
        self.num_users, self.num_items = interaction_mat.shape

        if timestamps is not None:
            self.timestamps = timestamps
            self.time_min = timestamps.min()
            self.time_max = timestamps.max()
        else:
            self.timestamps = None

        self.user_pos_items = {}
        for u in range(self.num_users):
            self.user_pos_items[u] = set(self.interaction_mat[u].indices.tolist())

    def get_user_timestamps(self, user_idx: int) -> np.ndarray:
        """Get sorted timestamps for a user's interactions."""
        if self.timestamps is None:
            return None
        items = self.interaction_mat[user_idx].indices
        mask = np.isin(np.arange(len(self.timestamps)), items)
        return self.timestamps[mask] if mask.any() else None

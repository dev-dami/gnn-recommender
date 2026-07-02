import torch
import torch.nn as nn
import numpy as np
from scipy import sparse


class NodeDropout(nn.Module):
    """Randomly zero out entire node embeddings during training."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand(x.size(0), 1, device=x.device) > self.p).float()
        return x * mask


class EdgeDropout(nn.Module):
    """Randomly drop edges from the sparse adjacency matrix during training.

    Each edge is independently dropped with probability p.
    Self-loops are never dropped to preserve node identity.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return adj

        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        size = adj.size()

        n_edges = values.size(0)
        keep_mask = torch.rand(n_edges, device=adj.device) >= self.p

        new_values = values * keep_mask.float()
        return torch.sparse_coo_tensor(indices, new_values, size)


class DropEdge:
    """Structural edge dropout that operates on scipy sparse matrices.

    Used during graph construction to create training-time augmented views.
    """

    def __init__(self, drop_rate: float = 0.1, seed: int = 42):
        self.drop_rate = drop_rate
        self.rng = np.random.RandomState(seed)

    def __call__(self, adj: sparse.csr_matrix) -> sparse.csr_matrix:
        if self.drop_rate == 0:
            return adj

        coo = adj.tocoo()
        n_edges = len(coo.data)
        keep = self.rng.random(n_edges) >= self.drop_rate

        dropped = sparse.csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=adj.shape,
        )
        return dropped

import torch
import torch.nn as nn
from scipy import sparse
import numpy as np


class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(norm_adj, embeddings)


class SparseDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand(x._nnz()) > self.p).to(x.dtype)
        return torch.sparse_coo_tensor(
            x._indices(),
            x._values() * mask,
            x.size(),
        )


def scipy_sparse_to_torch(sp_mat: sparse.csr_matrix) -> torch.Tensor:
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape)

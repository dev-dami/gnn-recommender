import numpy as np
from scipy import sparse


def build_adjacency_matrix(interaction_mat: sparse.csr_matrix) -> sparse.csr_matrix:
    num_users, num_items = interaction_mat.shape

    R = interaction_mat.astype(np.float32)
    zeros_ui = sparse.csr_matrix((num_users, num_users), dtype=np.float32)
    zeros_ii = sparse.csr_matrix((num_items, num_items), dtype=np.float32)

    adj = sparse.bmat([
        [zeros_ui, R],
        [R.T, zeros_ii],
    ], format="csr")

    return adj


def symmetric_normalization(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    d = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sparse.diags(d_inv_sqrt)
    norm = D @ adj @ D
    return norm.tocsr()


def compute_degree_distribution(adj: sparse.csr_matrix) -> np.ndarray:
    return np.array(adj.sum(axis=1)).flatten()

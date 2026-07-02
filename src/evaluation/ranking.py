import numpy as np
import torch
from scipy import sparse


def get_top_k_recommendations(user_emb: torch.Tensor, item_emb: torch.Tensor,
                               user_idx: int, k: int = 10,
                               train_mat: sparse.csr_matrix = None) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        scores = torch.mv(item_emb, user_emb[user_idx])
        if train_mat is not None:
            train_items = train_mat[user_idx].indices.tolist()
            scores[train_items] = float("-inf")
        topk = torch.topk(scores, k)
        return topk.indices.cpu().numpy(), topk.values.cpu().numpy()


def batch_recommendations(user_emb: torch.Tensor, item_emb: torch.Tensor,
                          user_indices: np.ndarray, k: int = 10,
                          train_mat: sparse.csr_matrix = None) -> dict:
    results = {}
    for u in user_indices:
        recs, scores = get_top_k_recommendations(user_emb, item_emb, int(u), k, train_mat)
        results[int(u)] = {"items": recs, "scores": scores}
    return results

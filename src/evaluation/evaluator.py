import numpy as np
import torch
from scipy import sparse

from .metrics import precision_at_k, recall_at_k, ndcg_at_k, map_at_k, hit_rate_at_k


class Evaluator:
    def __init__(self, test_mat: sparse.csr_matrix, train_mat: sparse.csr_matrix = None,
                 top_k: list = None):
        self.test_mat = test_mat.tocsr()
        self.train_mat = train_mat.tocsr() if train_mat is not None else None
        self.top_k = top_k or [10, 20]

        self.user_test_items = {}
        for u in range(test_mat.shape[0]):
            items = test_mat[u].indices.tolist()
            if items:
                self.user_test_items[u] = set(items)

    @torch.no_grad()
    def evaluate(self, user_emb: torch.Tensor, item_emb: torch.Tensor, k: int = 10) -> dict:
        num_users = user_emb.shape[0]
        all_metrics = {f"Precision@{k}": [], f"Recall@{k}": [],
                       f"NDCG@{k}": [], f"MAP@{k}": [], f"HitRate@{k}": []}

        for u in range(num_users):
            if u not in self.user_test_items:
                continue

            scores = torch.mv(item_emb, user_emb[u])

            if self.train_mat is not None:
                train_items = self.train_mat[u].indices.tolist()
                scores[train_items] = float("-inf")

            _, topk_indices = torch.topk(scores, k)
            recommended = topk_indices.cpu().numpy()
            relevant = self.user_test_items[u]

            all_metrics[f"Precision@{k}"].append(precision_at_k(recommended, relevant, k))
            all_metrics[f"Recall@{k}"].append(recall_at_k(recommended, relevant, k))
            all_metrics[f"NDCG@{k}"].append(ndcg_at_k(recommended, relevant, k))
            all_metrics[f"MAP@{k}"].append(map_at_k(recommended, relevant, k))
            all_metrics[f"HitRate@{k}"].append(hit_rate_at_k(recommended, relevant, k))

        results = {}
        for metric, values in all_metrics.items():
            results[metric] = np.mean(values) if values else 0.0

        return results

    def evaluate_all_k(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> dict:
        all_results = {}
        for k in self.top_k:
            results = self.evaluate(user_emb, item_emb, k)
            all_results.update(results)
        return all_results

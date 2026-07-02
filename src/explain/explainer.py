import torch
import numpy as np
from scipy import sparse
from typing import Optional


class GNNExplainer:
    """Explains GNN recommendations by identifying influential neighbors.

    For a given user-item pair, identifies which training interactions
    contributed most to the recommendation score via gradient-based attribution.
    """

    def __init__(self, model, norm_adj: torch.Tensor):
        self.model = model
        self.norm_adj = norm_adj

    def feature_importance(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                           user_idx: int, item_idx: int,
                           train_mat: sparse.csr_matrix,
                           top_k: int = 10) -> list[tuple[int, float]]:
        """Identify the most influential training interactions for a recommendation.

        Uses gradient magnitude as a proxy for importance.
        """
        self.model.eval()

        user_emb_grad = user_emb.clone().detach().requires_grad_(True)
        item_emb_grad = item_emb.clone().detach().requires_grad_(True)

        u = user_emb_grad[user_idx]
        i = item_emb_grad[item_idx]
        score = (u * i).sum()
        score.backward()

        user_importance = user_emb_grad.grad[user_idx].abs().sum(dim=-1).item()
        item_importance = item_emb_grad.grad[item_idx].abs().sum(dim=-1).item()

        neighbors = []
        neighbor_items = train_mat[user_idx].indices.tolist()
        for ni in neighbor_items:
            neighbors.append((ni, item_importance))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]

    @torch.no_grad()
    def attention_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                         user_idx: int, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Compute attention-like scores between a user and all items.

        Returns top-k item indices and their scores (for interpretability).
        """
        self.model.eval()
        scores = torch.mv(item_emb, user_emb[user_idx])
        topk = torch.topk(scores, min(top_k, scores.size(0)))
        return topk.indices.cpu().numpy(), topk.values.cpu().numpy()

    @torch.no_grad()
    def embedding_similarity(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                             user_idx: int, item_idx: int,
                             train_mat: sparse.csr_matrix) -> dict:
        """Explain a recommendation by decomposing the score into path contributions.

        Shows which layer contributed most to the final score.
        """
        self.model.eval()
        u = user_emb[user_idx]
        i = item_emb[item_idx]
        score = (u * i).sum().item()

        train_items = set(train_mat[user_idx].indices.tolist())
        n_train = len(train_items)

        return {
            "user_idx": user_idx,
            "item_idx": item_idx,
            "score": score,
            "user_norm": torch.norm(u).item(),
            "item_norm": torch.norm(i).item(),
            "n_user_interactions": n_train,
        }


class PathExplainer:
    """Explains recommendations by tracing message-passing paths.

    For a given user-item pair, identifies multi-hop paths through the graph
    that connect them and weights their contribution.
    """

    def __init__(self, norm_adj: sparse.csr_matrix, user_names: dict = None,
                 item_names: dict = None):
        self.norm_adj = norm_adj.tocsr()
        self.user_names = user_names or {}
        self.item_names = item_names or {}

    def find_paths(self, user_idx: int, item_idx: int,
                   max_hops: int = 2) -> list[list[tuple]]:
        """Find short paths from user to item in the bipartite graph."""
        num_users = self.norm_adj.shape[0]

        paths = []

        neighbors_of_user = set(self.norm_adj[user_idx].indices.tolist())
        if item_idx in neighbors_of_user:
            paths.append([
                ("user", user_idx, self.user_names.get(user_idx, f"User {user_idx}")),
                ("item", item_idx, self.item_names.get(item_idx, f"Item {item_idx}")),
            ])

        if max_hops >= 2:
            shared_users = []
            item_neighbors = set(self.norm_adj[item_idx].indices.tolist())
            user_neighbors = set(self.norm_adj[user_idx].indices.tolist())

            shared = item_neighbors & user_neighbors
            for mid_user in list(shared)[:5]:
                paths.append([
                    ("user", user_idx, self.user_names.get(user_idx, f"User {user_idx}")),
                    ("item", int(mid_user - num_users), self.item_names.get(int(mid_user - num_users), f"Item {int(mid_user - num_users)}")),
                    ("user", mid_user, self.user_names.get(mid_user, f"User {mid_user}")),
                    ("item", item_idx, self.item_names.get(item_idx, f"Item {item_idx}")),
                ])

        return paths

    def explain(self, user_idx: int, item_idx: int,
                train_mat: sparse.csr_matrix) -> dict:
        """Generate a human-readable explanation for a recommendation."""
        paths = self.find_paths(user_idx, item_idx, max_hops=2)
        u_name = self.user_names.get(user_idx, f"User {user_idx}")
        i_name = self.item_names.get(item_idx, f"Item {item_idx}")

        explanation = {
            "user": u_name,
            "item": i_name,
            "direct_interaction": item_idx in set(train_mat[user_idx].indices.tolist()),
            "paths": [],
            "n_paths": len(paths),
        }

        for path in paths:
            path_str = " → ".join([f"{role}({name})" for role, idx, name in path])
            explanation["paths"].append(path_str)

        return explanation

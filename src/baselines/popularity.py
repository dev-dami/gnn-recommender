import numpy as np
from scipy import sparse


class PopularityRecommender:
    def __init__(self, interaction_mat: sparse.csr_matrix):
        self.interaction_mat = interaction_mat.tocsr()
        self.item_popularity = np.array(interaction_mat.sum(axis=0)).flatten()

    def recommend(self, user_idx: int, k: int = 10, train_mat: sparse.csr_matrix = None) -> tuple[np.ndarray, np.ndarray]:
        scores = self.item_popularity.copy()
        if train_mat is not None:
            train_items = train_mat[user_idx].indices.tolist()
            scores[train_items] = -np.inf
        topk = np.argsort(scores)[-k:][::-1]
        return topk, scores[topk]

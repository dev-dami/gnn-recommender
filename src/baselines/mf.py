import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class MatrixFactorizationRecommender:
    def __init__(self, interaction_mat: sparse.csr_matrix, embedding_dim: int = 64):
        self.interaction_mat = interaction_mat.tocsr()
        self.embedding_dim = embedding_dim

    def fit(self):
        svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        self.user_emb = svd.fit_transform(self.interaction_mat)
        self.item_emb = svd.components_.T

    def recommend(self, user_idx: int, k: int = 10, train_mat: sparse.csr_matrix = None) -> tuple[np.ndarray, np.ndarray]:
        scores = self.item_emb @ self.user_emb[user_idx]
        if train_mat is not None:
            train_items = train_mat[user_idx].indices.tolist()
            scores[train_items] = -np.inf
        topk = np.argsort(scores)[-k:][::-1]
        return topk, scores[topk]

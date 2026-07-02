import numpy as np
from scipy import sparse


class InteractionDataset:
    def __init__(self, interaction_mat: sparse.csr_matrix):
        self.interaction_mat = interaction_mat.tocsr()
        self.user_items = {}
        for u in range(interaction_mat.shape[0]):
            self.user_items[u] = set(interaction_mat[u].indices.tolist())

    @property
    def num_users(self) -> int:
        return self.interaction_mat.shape[0]

    @property
    def num_items(self) -> int:
        return self.interaction_mat.shape[1]

    def get_pos_items(self, user_idx: int) -> np.ndarray:
        return np.array(list(self.user_items[user_idx]), dtype=np.int64)

    def build_user_item_map(self) -> dict:
        return self.user_items

import numpy as np
from scipy import sparse


class NegativeSampler:
    def __init__(self, interaction_mat: sparse.csr_matrix, num_negatives: int = 1, seed: int = 42):
        self.num_users, self.num_items = interaction_mat.shape
        self.interaction_mat = interaction_mat.tocsr()
        self.num_negatives = num_negatives
        self.rng = np.random.RandomState(seed)

        self.user_pos_items = {}
        for u in range(self.num_users):
            self.user_pos_items[u] = set(self.interaction_mat[u].indices.tolist())

    def sample(self, user_idx: int, pos_item_idx: int) -> list[int]:
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = self.rng.randint(0, self.num_items)
            if neg not in self.user_pos_items[user_idx]:
                negatives.append(neg)
        return negatives

    def sample_batch(self, user_indices: np.ndarray, pos_item_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_users = []
        batch_pos = []
        batch_neg = []

        for u, p in zip(user_indices, pos_item_indices):
            negs = self.sample(int(u), int(p))
            for n in negs:
                batch_users.append(u)
                batch_pos.append(p)
                batch_neg.append(n)

        return (
            np.array(batch_users, dtype=np.int64),
            np.array(batch_pos, dtype=np.int64),
            np.array(batch_neg, dtype=np.int64),
        )


class HardNegativeSampler(NegativeSampler):
    def __init__(self, interaction_mat: sparse.csr_matrix, num_negatives: int = 1,
                 popular_weight: float = 0.75, seed: int = 42):
        super().__init__(interaction_mat, num_negatives, seed)
        item_counts = np.array(interaction_mat.sum(axis=0)).flatten()
        self.item_probs = item_counts ** popular_weight
        self.item_probs /= self.item_probs.sum()

    def sample(self, user_idx: int, pos_item_idx: int) -> list[int]:
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = self.rng.choice(self.num_items, p=self.item_probs)
            if neg not in self.user_pos_items[user_idx]:
                negatives.append(neg)
        return negatives

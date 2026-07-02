import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse


class SelfSupervisedLoss(nn.Module):
    """Self-supervised contrastive learning for graph recommendations.

    Implements contrastive objectives that create augmented views of the
    user-item graph and learn to distinguish positive pairs from negative ones.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two sets of embeddings."""
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                      all_indices: torch.Tensor = None) -> torch.Tensor:
        """InfoNCE contrastive loss between two augmented views.

        Positive pairs: same node in different views.
        Negative pairs: different nodes.
        """
        sim_matrix = self.sim(z1, z2) / self.temperature
        n = z1.size(0)

        if all_indices is not None:
            labels = all_indices
        else:
            labels = torch.arange(n, device=z1.device)

        return F.cross_entropy(sim_matrix, labels)

    def bpr_ssl_loss(self, user_emb_1: torch.Tensor, user_emb_2: torch.Tensor,
                     item_emb_1: torch.Tensor, item_emb_2: torch.Tensor,
                     user_indices: torch.Tensor, pos_indices: torch.Tensor,
                     neg_indices: torch.Tensor) -> torch.Tensor:
        """Combined BPR + self-supervised loss.

        BPR loss on task-specific embeddings + contrastive loss on augmented views.
        """
        u1 = user_emb_1[user_indices]
        u2 = user_emb_2[user_indices]
        p1 = item_emb_1[pos_indices]
        p2 = item_emb_2[pos_indices]
        n1 = item_emb_1[neg_indices]
        n2 = item_emb_2[neg_indices]

        pos_score = (u1 * p1).sum(dim=-1) + (u2 * p2).sum(dim=-1)
        neg_score = (u1 * n1).sum(dim=-1) + (u2 * n2).sum(dim=-1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

        user_ssl = self.info_nce_loss(user_emb_1, user_emb_2)
        item_ssl = self.info_nce_loss(item_emb_1, item_emb_2)

        return bpr_loss + 0.1 * (user_ssl + item_ssl)


class GraphAugmentor:
    """Creates augmented views of the user-item graph for self-supervised learning.

    Strategies:
    - Edge perturbation: randomly add/remove edges
    - Node dropout: randomly mask node embeddings
    - Subgraph sampling: sample random subgraphs
    """

    def __init__(self, interaction_mat: sparse.csr_matrix, seed: int = 42):
        self.interaction_mat = interaction_mat.tocsr()
        self.num_users, self.num_items = interaction_mat.shape
        self.rng = np.random.RandomState(seed)

    def edge_perturbation(self, drop_rate: float = 0.1, add_rate: float = 0.05) -> sparse.csr_matrix:
        """Randomly drop existing edges and add new random edges."""
        coo = self.interaction_mat.tocoo()

        n_edges = len(coo.data)
        keep = self.rng.random(n_edges) >= drop_rate
        kept_rows, kept_cols = coo.row[keep], coo.col[keep]

        n_add = int(n_edges * add_rate)
        new_rows = self.rng.randint(0, self.num_users, n_add)
        new_cols = self.rng.randint(0, self.num_items, n_add)

        all_rows = np.concatenate([kept_rows, new_rows])
        all_cols = np.concatenate([kept_cols, new_cols])
        all_data = np.ones(len(all_rows), dtype=np.float32)

        return sparse.csr_matrix((all_data, (all_rows, all_cols)),
                                 shape=(self.num_users, self.num_items))

    def node_dropout(self, drop_rate: float = 0.1) -> sparse.csr_matrix:
        """Randomly zero out a fraction of all edges incident to dropped nodes."""
        mat = self.interaction_mat.copy().tolil()

        drop_users = self.rng.random(self.num_users) < drop_rate
        drop_items = self.rng.random(self.num_items) < drop_rate

        for u in np.where(drop_users)[0]:
            mat[u, :] = 0
        for i in np.where(drop_items)[0]:
            mat[:, i] = 0

        return mat.tocsr()

    def generate_view(self, strategy: str = "edge_perturbation", **kwargs) -> sparse.csr_matrix:
        """Generate a single augmented view."""
        if strategy == "edge_perturbation":
            return self.edge_perturbation(**kwargs)
        elif strategy == "node_dropout":
            return self.node_dropout(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

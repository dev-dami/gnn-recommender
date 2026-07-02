import numpy as np
from scipy import sparse
from collections import defaultdict


class KnowledgeGraph:
    """Lightweight knowledge graph for augmenting the user-item graph.

    Constructs edges between items based on shared genres, directors, or
    other metadata to provide semantic side information.
    """

    def __init__(self, num_items: int):
        self.num_items = num_items
        self.edges = []
        self.edge_types = defaultdict(list)

    def add_item_item_edges(self, item_a: int, item_b: int, edge_type: str = "genre"):
        """Add an undirected edge between two items."""
        self.edges.append((item_a, item_b, edge_type))
        self.edges.append((item_b, item_a, edge_type))
        self.edge_types[edge_type].append((item_a, item_b))

    def build_from_genre_matrix(self, genre_matrix: np.ndarray,
                                 threshold: float = 0.5):
        """Build item-item edges from genre overlap.

        Args:
            genre_matrix: (num_items × num_genres) binary matrix
            threshold: minimum Jaccard similarity to create an edge
        """
        n_items = genre_matrix.shape[0]
        for i in range(n_items):
            for j in range(i + 1, n_items):
                intersection = np.sum(genre_matrix[i] & genre_matrix[j])
                union = np.sum(genre_matrix[i] | genre_matrix[j])
                if union > 0 and intersection / union >= threshold:
                    self.add_item_item_edges(i, j, "genre")

    def build_from_interactions(self, interaction_mat: sparse.csr_matrix,
                                 co_occurrence_threshold: int = 5):
        """Build item-item edges from user co-occurrence.

        Items that appear together in many users' histories are connected.
        """
        mat = interaction_mat.tocsr()
        co_occurrence = (mat.T @ mat).toarray()
        np.fill_diagonal(co_occurrence, 0)

        for i in range(co_occurrence.shape[0]):
            for j in range(i + 1, co_occurrence.shape[1]):
                if co_occurrence[i, j] >= co_occurrence_threshold:
                    self.add_item_item_edges(i, j, "co_occurrence")

    def get_augmented_adjacency(self, original_adj: sparse.csr_matrix,
                                 kg_weight: float = 0.5) -> sparse.csr_matrix:
        """Merge user-item graph with knowledge graph edges.

        Returns a combined adjacency matrix where KG edges are weighted
        by `kg_weight` relative to interaction edges.
        """
        n_total = original_adj.shape[0]

        if not self.edges:
            return original_adj

        rows = []
        cols = []
        data = []

        orig_coo = original_adj.tocoo()
        rows.extend(orig_coo.row.tolist())
        cols.extend(orig_coo.col.tolist())
        data.extend(orig_coo.data.tolist())

        for item_a, item_b, _ in self.edges:
            row_a = n_total - self.num_items + item_a
            row_b = n_total - self.num_items + item_b
            rows.extend([row_a, row_b])
            cols.extend([row_b, row_a])
            data.extend([kg_weight, kg_weight])

        aug_adj = sparse.csr_matrix(
            (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
            shape=(n_total, n_total),
        )
        return aug_adj

    def get_item_neighbors(self, item_idx: int) -> dict:
        """Get all knowledge-graph neighbors of an item."""
        neighbors = defaultdict(list)
        for a, b, etype in self.edges:
            if a == item_idx:
                neighbors[etype].append(b)
        return dict(neighbors)

    @property
    def num_edges(self) -> int:
        return len(self.edges) // 2

    def summary(self) -> dict:
        type_counts = {k: len(v) // 2 for k, v in self.edge_types.items()}
        return {
            "total_edges": self.num_edges,
            "edge_types": type_counts,
        }

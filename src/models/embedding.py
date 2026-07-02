import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_embedding.weight, self.item_embedding.weight

    def get_user_embedding(self, user_indices: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(user_indices)

    def get_item_embedding(self, item_indices: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_indices)

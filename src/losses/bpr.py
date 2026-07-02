import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        diff = pos_scores - neg_scores
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-8))
        return loss

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses.bpr import BPRLoss


def test_bpr_loss_basic():
    loss_fn = BPRLoss()
    pos_scores = torch.tensor([1.0, 2.0, 3.0])
    neg_scores = torch.tensor([0.1, 0.5, 1.0])
    loss = loss_fn(pos_scores, neg_scores)
    assert loss.item() > 0
    assert loss.item() < 5


def test_bpr_loss_perfect():
    loss_fn = BPRLoss()
    pos_scores = torch.tensor([10.0, 10.0])
    neg_scores = torch.tensor([0.0, 0.0])
    loss = loss_fn(pos_scores, neg_scores)
    assert loss.item() < 0.1


def test_bpr_loss_backward():
    loss_fn = BPRLoss()
    pos_scores = torch.tensor([1.0, 2.0], requires_grad=True)
    neg_scores = torch.tensor([0.1, 0.5], requires_grad=True)
    loss = loss_fn(pos_scores, neg_scores)
    loss.backward()
    assert pos_scores.grad is not None
    assert neg_scores.grad is not None


if __name__ == "__main__":
    test_bpr_loss_basic()
    test_bpr_loss_perfect()
    test_bpr_loss_backward()
    print("All BPR loss tests passed!")

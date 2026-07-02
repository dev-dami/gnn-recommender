import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, map_at_k, hit_rate_at_k


def test_precision_at_k():
    recommended = np.array([1, 2, 3, 4, 5])
    relevant = {1, 3, 5}
    assert abs(precision_at_k(recommended, relevant, 5) - 3 / 5) < 1e-6
    assert abs(precision_at_k(recommended, relevant, 3) - 2 / 3) < 1e-6


def test_recall_at_k():
    recommended = np.array([1, 2, 3, 4, 5])
    relevant = {1, 3, 5, 7, 9}
    assert abs(recall_at_k(recommended, relevant, 5) - 3 / 5) < 1e-6


def test_ndcg_at_k():
    recommended = np.array([1, 2, 3])
    relevant = {1, 2}
    ndcg = ndcg_at_k(recommended, relevant, 3)
    assert 0 <= ndcg <= 1


def test_hit_rate():
    recommended = np.array([1, 2, 3])
    relevant = {1}
    assert hit_rate_at_k(recommended, relevant, 3) == 1.0
    relevant2 = {10}
    assert hit_rate_at_k(recommended, relevant2, 3) == 0.0


def test_map_at_k():
    recommended = np.array([1, 2, 3, 4, 5])
    relevant = {1, 3, 5}
    map_score = map_at_k(recommended, relevant, 5)
    assert 0 <= map_score <= 1


if __name__ == "__main__":
    test_precision_at_k()
    test_recall_at_k()
    test_ndcg_at_k()
    test_hit_rate()
    test_map_at_k()
    print("All metric tests passed!")

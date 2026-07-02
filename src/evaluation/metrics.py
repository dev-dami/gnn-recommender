import numpy as np


def precision_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / k


def recall_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))

    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant) if len(relevant) > 0 else 0.0


def hit_rate_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    return 1.0 if len(set(recommended_k) & relevant) > 0 else 0.0

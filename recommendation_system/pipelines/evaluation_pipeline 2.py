import numpy as np

from recommendation_system.utils.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def evaluate_single(
    recommended: list[str],
    relevant: list[str],
    relevance_scores: list[float],
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate a single recommendation list.

    Parameters
    ----------
    recommended : List[str]
    relevant : List[str]
    relevance_scores : List[float]
    k : int

    Returns
    -------
    Dict[str, float]
        A dictionary containing the evaluated results.
    """
    return {
        "precision@k": precision_at_k(recommended, relevant, k),
        "recall@k": recall_at_k(recommended, relevant, k),
        "hit_rate@k": hit_rate_at_k(recommended, relevant, k),
        "ndcg@k": ndcg_at_k(relevance_scores, k),
    }


def evaluate_dataset(
    all_recommendations: list[list[str]],
    all_relevants: list[list[str]],
    all_relevance_scores: list[list[float]],
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate a full dataset.

    Returns mean metrics.
    """
    metrics = []

    for rec, rel, rel_scores in zip(
        all_recommendations,
        all_relevants,
        all_relevance_scores,
        strict=False,
    ):
        metrics.append(evaluate_single(rec, rel, rel_scores, k))

    results = {}

    for key in metrics[0].keys():
        results[key] = float(np.mean([m[key] for m in metrics]))

    return results

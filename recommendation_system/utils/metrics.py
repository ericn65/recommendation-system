from collections.abc import Iterable, Sequence

import numpy as np


def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    Compute Precision@K.

    Parameters
    ----------
    recommended : List[str]
        Ranked list of recommended items.
    relevant : List[str]
        List of relevant (ground truth) items.
    k : int
        Number of top elements to consider.

    Returns
    -------
    float
        Precision at K.
    """
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    Compute Recall@K.

    Parameters
    ----------
    recommended : List[str]
        Ranked list of recommended items.
    relevant : List[str]
        List of relevant (ground truth) items.
    k : int
        Number of top elements to consider.

    Returns
    -------
    float
        Recall at K.
    """
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def hit_rate_at_k(recommended: list[str], relevant: list[str], k: int) -> int:
    """
    Compute Hit Rate@K.

    Parameters
    ----------
    recommended : List[str]
        Ranked list of recommended items.
    relevant : List[str]
        List of relevant items.
    k : int
        Number of top elements to consider.

    Returns
    -------
    int
        1 if any relevant item appears in top K, else 0.
    """
    recommended_k = recommended[:k]
    return int(len(set(recommended_k) & set(relevant)) > 0)


def average_precision(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    Compute Average Precision (AP) at K.

    Parameters
    ----------
    recommended : List[str]
        Ranked list of recommended items.
    relevant : List[str]
        Ground truth relevant items.
    k : int
        Number of top elements.

    Returns
    -------
    float
        Average precision score.
    """
    score = 0.0
    hits = 0

    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i

    return score / len(relevant) if relevant else 0.0


def mean_average_precision(
    all_recommended: Iterable[list[str]],
    all_relevant: Iterable[list[str]],
    k: int,
) -> float:
    """
    Compute Mean Average Precision (MAP).

    Parameters
    ----------
    all_recommended : Iterable[List[str]]
        List of recommendation lists.
    all_relevant : Iterable[List[str]]
        List of ground truth relevant lists.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        MAP score.
    """
    scores = [
        average_precision(rec, rel, k)
        for rec, rel in zip(all_recommended, all_relevant, strict=False)
    ]

    return float(np.mean(scores)) if scores else 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain (DCG).

    Parameters
    ----------
    relevances : List[float]
        Relevance scores in ranked order.
    k : int
        Rank cutoff.

    Returns
    -------
    float
        DCG score.
    """
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Parameters
    ----------
    relevances : List[float]
        Relevance scores for the predicted ranking.
    k : int
        Rank cutoff.

    Returns
    -------
    float
        NDCG score between 0 and 1.
    """
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def coverage(recommended_items: Iterable[list[str]], catalog: list[str]) -> float:
    """
    Compute catalog coverage.

    Parameters
    ----------
    recommended_items : Iterable[List[str]]
        Recommendations for all users.
    catalog : List[str]
        Full item catalog.

    Returns
    -------
    float
        Fraction of catalog items recommended.
    """
    recommended_set = set()

    for rec_list in recommended_items:
        recommended_set.update(rec_list)

    return len(recommended_set) / len(catalog) if catalog else 0.0


def diversity(recommended: list[str], similarity_matrix: dict) -> float:
    """
    Compute diversity score based on item similarity.

    Parameters
    ----------
    recommended : List[str]
        List of recommended items.
    similarity_matrix : dict
        Dict[item1][item2] giving similarity score.

    Returns
    -------
    float
        Diversity score (higher means more diverse).
    """
    if len(recommended) < 2:
        return 0.0

    sims = []

    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            sims.append(
                similarity_matrix.get(recommended[i], {}).get(recommended[j], 0)
            )

    return 1 - np.mean(sims) if sims else 0.0

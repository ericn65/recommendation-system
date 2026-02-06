from typing import Any

import pandas as pd

from recommendation_system.metrics.protocols import MetricProtocol


class PrecisionAtK(MetricProtocol):
    """Precision@K metric. Class which inherates from the protocol."""

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        """
        Computation interface.

        Parameters
        ----------
        recommendations : dict[Any, list[Any]]
            The recommendations to evaluate.
        groun_truth : pd.DataFrame
            The results "labels" validated.
        k : int
            The number of recommendations that can appear.
        """
        precisions: list[float] = []

        gt = ground_truth.groupby("user_id")["item_id"].apply(set).to_dict()

        for user_id, recs in recommendations.items():
            if user_id not in gt:
                continue

            hits = len(set(recs[:k]) & gt[user_id])
            precisions.append(hits / k)

        return sum(precisions) / len(precisions) if precisions else 0.0

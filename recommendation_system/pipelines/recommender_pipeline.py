from collections.abc import Iterable
from typing import Any

import pandas as pd

from recommendation_system.metrics.protocols import MetricProtocol
from recommendation_system.recommenders.protocols import RecommenderProtocol


class RecommenderPipeline:
    """
    End-to-end recommender pipeline.

    Parameters
    ----------
    model : RecommenderProtocol
        The protocol for recomendation to add.
    metrics : Iterable[MetricProtocol]
        The protcol to measure metrics added.
    """

    def __init__(
        self,
        model: RecommenderProtocol,
        metrics: Iterable[MetricProtocol] | None = None,
    ) -> None:
        """Initializes the model to train and recommend."""
        self.model = model
        self.metrics = list(metrics) if metrics else []

    def fit(self, interactions: pd.DataFrame, **kwargs: Any) -> None:
        """
        Trains the model for recommendations.

        Parameters
        ----------
        interactions : pd.DataFrame
            All the interactions to train the model with.
        """
        self.model.fit(interactions, **kwargs)

    def recommend(self, users: Iterable[Any], k: int = 10) -> dict[Any, list[Any]]:
        """
        Compute relevance score for a user-item pair.

        Parameters
        ----------
        user_id : Any
            The unique identifier for the user who wants to get a
            recommendation.
        item_id : Any
            The item to recommend.

        Returns
        -------
        score : float or None
        """
        recommendations: dict[Any, list[Any]] = {}

        for user_id in users:
            recs = self.model.recommend(user_id, k)
            recommendations[user_id] = [item for item, _ in recs]

        return recommendations

    def evaluate(
        self, users: Iterable[Any], ground_truth: pd.DataFrame, k: int = 10
    ) -> dict[str, float]:
        """
        Evaluates the whole model checking with a ground-truth
        the capacity of the model to recommend stuff.

        Parameters
        ----------
        users : Iterable
            All the users to evaluate the system.
        ground_truth : pd.DataFrame
            The result of interaction used as a ground truth.
        k : int
            The number of recommendation to evaluate (top-k)
        """
        recs = self.recommend(users, k)

        results: dict[str, float] = {}
        for metric in self.metrics:
            name = metric.__class__.__name__
            results[name] = metric.compute(recs, ground_truth, k)

        return results

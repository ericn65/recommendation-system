from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class RecommenderProtocol(Protocol):
    """Structural typing for recommenders."""

    def fit(self, interactions: pd.DataFrame, **kwargs: Any | None) -> None:
        """
        Parameters
        ----------
        interactions : pd.DataFrame
            Columns: [user_id, item_id, weight]
        item_features : pd.DataFrame
            Columns: [item_id, f1, f2, ...]
        normalize : bool
            Normalizes the values or not according to the scope.
        """
        ...

    def score(self, user_id: Any, item_id: Any) -> float | None:
        """
        Scoring system for the content based recommender.

        Parameters
        ----------
        user_id : Any
            The identifier for the user.
        item_id : Any
            The item to recommend for.

        Returns
        -------
        score : float, optional
            The score for the process.
        """
        ...

    def recommend(self, user_id: Any, k: int = 10) -> list[tuple[Any, float]]:
        """
        Gets the most optimal recommendation for the user and its top options.

        Parameters
        ----------
        user_id : any
            The unique identifier for the user.
        k : int
            The number of recommendations given.

        Returns
        -------
        recommendations : list of (item_id, score)
        """
        ...

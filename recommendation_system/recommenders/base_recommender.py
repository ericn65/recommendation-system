from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseRecommender(ABC):
    """
    Base interface for all recommenders.

    Methods
    -------
    fit -> Trains the model
    recommend -> Returns top-k items for the user.
    score -> Scores the relevance of the prediction.
    """

    @abstractmethod
    def fit(
        self,
        interactions: pd.DataFrame,
        **kwargs: Any | None,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        interactions : pd.DataFrame
            The interactions made to train the model.
        **kwargs : any
            The different hyperparameters for the model.
        """
        pass

    @abstractmethod
    def score(self, user_id: Any, item_id: Any) -> float | None:
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
        pass

    @abstractmethod
    def recommend(self, user_id: Any, k: int = 10) -> list[tuple[Any, float]]:
        """
        Recommend top-k items for a user.

        Returns
        -------
        recommendations : list of (item_id, score)
        """
        pass

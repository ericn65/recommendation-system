from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommendation_system.recommenders.base_recommender import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommender using precomputed item feature vectors.

    It has no init and inherates from BaseRecommender.
    """

    def fit(self, interactions: pd.DataFrame, **kwargs: Any) -> None:
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
        item_features: pd.DataFrame = kwargs["item_features"]
        normalize: bool = kwargs.get("normalize", True)

        self.item_ids = item_features["item_id"].to_numpy()

        self.X: npt.NDArray[np.float64] = (
            item_features.drop(columns=["item_id"]).to_numpy().astype(np.float64)
        )

        if normalize:
            norm = np.linalg.norm(self.X, axis=1, keepdims=True)
            self.X = self.X / (norm + 1e-8)

        self.user_profiles: dict[Any, npt.NDArray[np.float64]] = {}

        for user_id, group in interactions.groupby("user_id"):
            indices = [
                int(np.where(self.item_ids == item)[0][0]) for item in group["item_id"]
            ]
            weights = group["weight"].to_numpy().reshape(-1, 1)
            self.user_profiles[user_id] = np.mean(self.X[indices] * weights, axis=0)

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
        if user_id not in self.user_profiles:
            return None

        idx = int(np.where(self.item_ids == item_id)[0][0])
        score: float = float(
            cosine_similarity(
                self.user_profiles[user_id].reshape(1, -1), self.X[idx].reshape(1, -1)
            )[0, 0]
        )
        return score

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
        if user_id not in self.user_profiles:
            raise ValueError("Unknown user")

        scores = cosine_similarity(
            self.user_profiles[user_id].reshape(1, -1), self.X
        ).flatten()

        top_idx = scores.argsort()[::-1][:k]
        return [(self.item_ids[i], float(scores[i])) for i in top_idx]

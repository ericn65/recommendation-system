from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from recommendation_system.recommenders.base_recommender import BaseRecommender


class MatrixFactorizationRecommender(BaseRecommender):
    """
    Matrix Factorization recommender using precomputed item feature vectors.

    It has no init and inherates from BaseRecommender.

    Parameters
    ----------
    n_factors : int
        The number of factors needed to compute the matrix.
    lr : float
        The learning rate to train the factorization.
    reg : float
        The capacity of regression to train.
    epochs : int
        The number of repetitions.
    """

    def __init__(
        self, n_factors: int = 32, lr: float = 0.01, reg: float = 0.1, epochs: int = 20
    ) -> None:
        """Initialization of the procedure."""
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

    def fit(self, interactions: pd.DataFramem, **kwargs: Any | None) -> None:
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
        self.users = interactions["user_id"].unique()
        self.items = interactions["item_id"].unique()

        self.u_map: dict[Any, int] = {u: i for i, u in enumerate(self.users)}
        self.i_map: dict[Any, int] = {i: j for j, i in enumerate(self.items)}

        self.P: npt.NDArray[np.float64] = np.random.normal(
            0, 0.1, (len(self.users), self.n_factors)
        )
        self.Q: npt.NDArray[np.float64] = np.random.normal(
            0, 0.1, (len(self.items), self.n_factors)
        )

        for _ in range(self.epochs):
            for row in interactions.itertuples():
                u = self.u_map[row.user_id]
                i = self.i_map[row.item_id]
                r = float(row.weight)

                pred = self.P[u] @ self.Q[i]
                err = r - pred

                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])

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
        if user_id not in self.u_map or item_id not in self.i_map:
            return None
        return float(self.P[self.u_map[user_id]] @ self.Q[self.i_map[item_id]])

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
        if user_id not in self.u_map:
            raise ValueError("Unknown user")

        u = self.u_map[user_id]
        scores = self.Q @ self.P[u]
        top = scores.argsort()[::-1][:k]

        return [(self.items[i], float(scores[i])) for i in top]

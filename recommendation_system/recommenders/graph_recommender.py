from typing import Any

import networkx as nx
import pandas as pd

from recommendation_system.recommenders.base_recommender import BaseRecommender


class GraphRecommender(BaseRecommender):
    """
    Graph-based recommender using personalized PageRank.

    Parameters
    ----------
    alpha : float
        The teleporting method.
    """

    def __init__(self, alpha: float = 0.85) -> None:
        """Initializes the model."""
        self.alpha = alpha

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
        self.G: nx.Graph = nx.Graph()

        for row in interactions.itertuples():
            u = f"u_{row.user_id}"
            i = f"i_{row.item_id}"
            self.G.add_edge(u, i, weight=float(row.weight))

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
        user_node = f"u_{user_id}"
        item_node = f"i_{item_id}"

        if user_node not in self.G or item_node not in self.G:
            return None

        scores = nx.pagerank(
            self.G, personalization={user_node: 1.0}, alpha=self.alpha, weight="weight"
        )
        return float(scores.get(item_node, 0.0))

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
        user_node = f"u_{user_id}"
        if user_node not in self.G:
            raise ValueError("Unknown user")

        scores = nx.pagerank(
            self.G, personalization={user_node: 1.0}, alpha=self.alpha, weight="weight"
        )

        items: dict[Any, float] = {
            n.replace("i_", ""): float(s)
            for n, s in scores.items()
            if n.startswith("i_")
        }

        return sorted(items.items(), key=lambda x: x[1], reverse=True)[:k]

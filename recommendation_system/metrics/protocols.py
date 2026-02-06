from typing import Any, Protocol

import pandas as pd


class MetricProtocol(Protocol):
    """Interface for evaluation metrics."""

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
        ...

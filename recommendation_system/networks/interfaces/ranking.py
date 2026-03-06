from typing import Protocol

from recommendation_system.networks.interfaces.retreival import CandidateList, ItemId


class PointwiseRanker(Protocol):
    """Model that scores each candidate independently."""

    def predict(self, item: ItemId, rival: ItemId) -> float:
        """Abstract method to predict with PointWise."""
        ...


class PairwiseRanker(Protocol):
    """Model trained with pairwise comparisons."""

    def score(self, item: ItemId, rival: ItemId) -> float:
        """Abstract method to score with PairWise."""
        ...


class LambdaRankModel(Protocol):
    """LambdaRank / LambdaMART style models."""

    def rank_items(
        self,
        candidates: CandidateList,
        rival: ItemId,
    ) -> list[tuple[ItemId, float]]:
        """Abstract method to rank items."""
        ...


class ListwiseRanker(Protocol):
    """Listwise ranking models (NN or GBDT)."""

    def predict_list(
        self,
        candidates: CandidateList,
        rival: ItemId,
    ) -> CandidateList:
        """Abstract method to predict lists."""
        ...

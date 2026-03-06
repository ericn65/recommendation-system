from collections.abc import Callable

from enums import RankingMethod

from recommendation_system.networks.interfaces.ranking import (
    LambdaRankModel,
    ListwiseRanker,
    PairwiseRanker,
    PointwiseRanker,
)
from recommendation_system.networks.interfaces.retreival import CandidateList, ItemId


def pointwise_rank(
    model: PointwiseRanker,
    candidates: CandidateList,
    rival: ItemId,
) -> CandidateList:
    """Rank candidates using a pointwise scoring model."""
    scored: list[tuple[ItemId, float]] = [
        (c, model.predict(c, rival)) for c in candidates
    ]

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    return [item for item, _ in ranked]


def pairwise_rank(
    model: PairwiseRanker,
    candidates: CandidateList,
    rival: ItemId,
) -> CandidateList:
    """Rank candidates using a pairwise ranking model."""
    scored: list[tuple[ItemId, float]] = [
        (c, model.score(c, rival)) for c in candidates
    ]

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    return [item for item, _ in ranked]


def lambdarank(
    model: LambdaRankModel,
    candidates: CandidateList,
    rival: ItemId,
) -> CandidateList:
    """LambdaRank / LambdaMART ranking."""
    scored = model.rank_items(candidates, rival)

    sorted_items = sorted(scored, key=lambda x: x[1], reverse=True)

    return [item for item, _ in sorted_items]


def listwise_rank(
    model: ListwiseRanker,
    candidates: CandidateList,
    rival: ItemId,
) -> CandidateList:
    """Listwise ranking models."""
    return model.predict_list(candidates, rival)


RankingFunction = Callable[..., CandidateList]


RANKING_DISPATCHER: dict[RankingMethod, RankingFunction] = {
    RankingMethod.POINTWISE: pointwise_rank,
    RankingMethod.PAIRWISE: pairwise_rank,
    RankingMethod.LAMBDARANK: lambdarank,
    RankingMethod.LISTWISE: listwise_rank,
}

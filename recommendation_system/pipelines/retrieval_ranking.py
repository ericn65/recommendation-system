from typing import cast

from recommendation_system.networks.interfaces.retreival import (
    ANNEmbeddingModel,
    CandidateList,
    Embedding,
    ItemId,
)
from recommendation_system.utils.enums import RankingMethod, RetrievalMethod
from recommendation_system.utils.raking import RANKING_DISPATCHER, RankingFunction
from recommendation_system.utils.retrieval import (
    RETRIEVAL_DISPATCHER,
    RetrievalFunction,
)


def recommend(
    team: list[ItemId],
    rival: ItemId,
    retrieval_method: RetrievalMethod,
    ranking_method: RankingMethod,
    retrieval_model: object,
    ranking_model: object,
    top_k_retrieval: int = 50,
    top_k_final: int = 6,
) -> CandidateList:
    """Two-stage recommendation pipeline (Retrieval → Ranking)."""
    # --- Retrieval stage ---
    retriever: RetrievalFunction = RETRIEVAL_DISPATCHER[retrieval_method]

    if retrieval_method == RetrievalMethod.ANN_EMBEDDINGS:
        ann_model = cast(ANNEmbeddingModel, retrieval_model)

        embedding: Embedding = ann_model.get_embedding(rival)

        candidates: CandidateList = retriever(
            ann_model.index,
            embedding,
            top_k_retrieval,
        )

    else:
        candidates = retriever(
            retrieval_model,
            rival,
            top_k_retrieval,
        )

    # Remove already owned items
    filtered_candidates: CandidateList = [c for c in candidates if c not in team]

    # --- Ranking stage ---
    ranker: RankingFunction = RANKING_DISPATCHER[ranking_method]

    final_ranked: CandidateList = ranker(
        ranking_model,
        filtered_candidates,
        rival,
    )

    return final_ranked[:top_k_final]

from recommendation_system.utils.enums import RankingMethod, RetrievalMethod
from recommendation_system.utils.raking import RANKING_DISPATCHER
from recommendation_system.utils.retrieval import RETRIEVAL_DISPATCHER


def recommend(
    team: list[str],
    rival: str,
    retrieval_method: RetrievalMethod,
    ranking_method: RankingMethod,
    retrieval_model: any,
    ranking_model: any,
    top_k_retrieval: int = 50,
    top_k_final: int = 6,
) -> list[str]:
    """
    Two-stage recommendation pipeline (Retrieval → Ranking).

    Parameters
    ----------
    team : List[str]
        Items the user owns (e.g., Pokémon).
    rival : str
        Target item to compare against.
    retrieval_method : RetrievalMethod
        Enum specifying retrieval algorithm.
    ranking_method : RankingMethod
        Enum specifying ranking algorithm.
    retrieval_model : Any
        Model to use for retrieval.
    ranking_model : Any
        Model to use for ranking.
    top_k_retrieval : int
        Number of candidates to retrieve.
    top_k_final : int
        Number of final recommendations.

    Returns
    -------
    List[str]
        Final ranked recommendations.
    """
    # --- Retrieval ---
    retriever = RETRIEVAL_DISPATCHER[retrieval_method]

    if retrieval_method.name == "ANN_EMBEDDINGS":
        embedding = retrieval_model.get_embedding(rival)
        candidates = retriever(retrieval_model.index, embedding, top_k_retrieval)
    else:
        candidates = retriever(retrieval_model, rival, top_k_retrieval)

    filtered_candidates = [c for c in candidates if c in team]

    ranker = RANKING_DISPATCHER[ranking_method]
    final_ranked = ranker(ranking_model, filtered_candidates, rival)

    return final_ranked[:top_k_final]

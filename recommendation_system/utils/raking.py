from enums import RankingMethod


def pointwise_rank(model: any, candidates: list[str], rival: str) -> list[str]:
    """
    Rank candidates using a pointwise scoring neural network or ML model.

    Parameters
    ----------
    model : Any
        Model with method .predict(item, rival)
    candidates : List[str]
        List of candidate items to sort.
    rival : str
        Target item to rank against.

    Returns
    -------
    List[str]
        Sorted list (best first).
    """
    scored = [(c, model.predict(c, rival)) for c in candidates]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked]


def pairwise_rank(model: any, candidates: list[str], rival: str) -> list[str]:
    """
    Rank candidates using a pairwise ranking model such as RankNet.

    Parameters
    ----------
    model : Any
        Must implement score(item, rival)
    candidates : List[str]
    rival : str

    Returns
    -------
    List[str]
        Sorted candidates.
    """
    scored = [(c, model.score(c, rival)) for c in candidates]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked]


def lambdarank(model: any, candidates: list[str], rival: str) -> list[str]:
    """
    LambdaRank-based ranking.

    Parameters
    ----------
    model : Any
        LambdaRank or LambdaMART model.
    """
    scored = model.rank_items(candidates, rival)
    sorted_items = sorted(scored, key=lambda x: x[1], reverse=True)
    return [p for p, s in sorted_items]


def listwise_rank(model: any, candidates: list[str], rival: str) -> list[str]:
    """
    Listwise NN or LambdaMART ranking.

    Parameters
    ----------
    model : Any
        Model must implement method: model.predict_list(candidates, rival)
    """
    scored = model.predict_list(candidates, rival)
    return [x for x in scored]


# Dispatcher
RANKING_DISPATCHER: dict[RankingMethod, callable] = {
    RankingMethod.POINTWISE: pointwise_rank,
    RankingMethod.PAIRWISE: pairwise_rank,
    RankingMethod.LAMBDARANK: lambdarank,
    RankingMethod.LISTWISE: listwise_rank,
}

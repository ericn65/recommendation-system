from collections.abc import Callable

import numpy as np

from recommendation_system.networks.interfaces.retreival import (
    ANNIndex,
    CandidateList,
    Embedding,
    ItemId,
    MFModel,
    Node2VecModel,
)
from recommendation_system.utils.enums import RetrievalMethod


def content_based_retrieval(
    vector_db: dict[ItemId, Embedding],
    rival: ItemId,
    top_k: int = 50,
) -> CandidateList:
    """
    Retrieve candidates based on cosine similarity.

    Parameters
    ----------
    vector_db : dict[ItemId, Embedding]
        Mapping from item_id to embedding vector.
    rival : ItemId
        Query item.
    top_k : int
        Number of candidates to retrieve.

    Returns
    -------
    CandidateList
        Top-k most similar items.
    """
    query_vec = vector_db[rival]
    sims: dict[ItemId, float] = {}

    for item, vec in vector_db.items():
        if item == rival:
            continue

        sims[item] = float(
            np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        )

    return sorted(sims, key=lambda item: sims[item], reverse=True)[:top_k]


def mf_retrieval(
    model: MFModel,
    rival: ItemId,
    top_k: int = 50,
) -> CandidateList:
    """Retrieve candidates using a Matrix Factorization model."""
    return model.get_similar_items(rival, top_k=top_k)


def node2vec_retrieval(
    model: Node2VecModel,
    rival: ItemId,
    top_k: int = 50,
) -> CandidateList:
    """Retrieve candidates using Node2Vec similarity."""
    sims = model.most_similar(rival, top_k)
    return [node for node, _ in sims]


def ann_retrieval(
    index: ANNIndex,
    embedding: Embedding,
    top_k: int = 50,
) -> list[int]:
    """Retrieve nearest neighbors using ANN index (e.g., FAISS)."""
    distances, ids = index.search(np.array([embedding]), top_k)
    return list(ids[0])


RETRIEVAL_DISPATCHER: dict[RetrievalMethod, Callable] = {
    RetrievalMethod.CONTENT_BASED: content_based_retrieval,
    RetrievalMethod.MATRIX_FACTORIZATION: mf_retrieval,
    RetrievalMethod.NODE2VEC: node2vec_retrieval,
    RetrievalMethod.ANN_EMBEDDINGS: ann_retrieval,
}

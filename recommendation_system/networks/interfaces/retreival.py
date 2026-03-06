from typing import Protocol, TypeAlias

import numpy as np

ItemId: TypeAlias = str
Embedding: TypeAlias = np.ndarray
CandidateList: TypeAlias = list[ItemId]


class MFModel(Protocol):
    """Interface for Matrix Factorization retrieval models."""

    def get_similar_items(self, item_id: ItemId, top_k: int) -> CandidateList:
        """Return top-k items similar to the given item."""
        ...


class Node2VecModel(Protocol):
    """Interface for Node2Vec graph embedding models."""

    def most_similar(self, node: ItemId, top_k: int) -> list[tuple[ItemId, float]]:
        """Return the top-k most similar nodes and their similarity score."""
        ...


class ANNIndex(Protocol):
    """Interface for Approximate Nearest Neighbor indexes."""

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search nearest neighbors.

        Returns
        -------
        distances : np.ndarray
        ids : np.ndarray
        """
        ...


class ANNEmbeddingModel(Protocol):
    """Model providing embeddings for ANN retrieval."""

    index: ANNIndex

    def get_embedding(self, item: ItemId) -> Embedding:
        """Middle interface to ensure interconnection."""
        ...


class RetrievalModel(Protocol):
    """Generic retrieval interface."""

    def retrieve(self, query: ItemId, top_k: int) -> CandidateList:
        """Generic retrieve item."""
        ...

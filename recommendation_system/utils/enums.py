from enum import StrEnum, auto


class DataTypesEnum(StrEnum):
    """Available data types to extract."""

    CSV = auto()
    JSON = auto()
    SQL = auto()
    EXCEL = auto()
    OTHER = auto()


class RetrievalMethod(StrEnum):
    """Types of retrieval analysis allowed in our system."""

    CONTENT_BASED = auto()
    MATRIX_FACTORIZATION = auto()
    NODE2VEC = auto()
    ANN_EMBEDDINGS = auto()


class RankingMethod(StrEnum):
    """Types of ranking methodologies allowed in our systems."""

    POINTWISE = auto()
    PAIRWISE = auto()
    LAMBDARANK = auto()
    LISTWISE = auto()

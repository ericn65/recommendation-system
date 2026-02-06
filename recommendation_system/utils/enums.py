from enum import Enum, auto


class DataTypesEnum(Enum):
    """Available data types to extract."""

    CSV = auto()
    JSON = auto()
    SQL = auto()
    EXCEL = auto()
    OTHER = auto()

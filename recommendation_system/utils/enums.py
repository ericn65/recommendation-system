from enum import Enum, StrEnum, auto


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


class DataTypeEnum(StrEnum):
    """The types of input data for the system."""

    CSV = auto()
    EXCEL = auto()
    SQL = auto()
    JSON = auto()


class SpecificRecommendationEnum(Enum):
    """The specific csv to use as data in any case."""

    USERS = "users.csv"
    ITEMS = "items.csv"
    INTERACTIONS = "interactions.csv"


class FeatureNamesEnum(StrEnum):
    """The type of feature names accepted in the recommendation system."""

    STRESS = auto()
    BURNOUT = auto()

    MOTIVATION = auto()
    RESILIENCE = auto()
    OPTIMISM = auto()
    SELF_CONFIDANCE = auto()
    FULFILMENT = auto()
    COMMITMENT = auto()
    AUTONOMY = auto()

    GENERAL_WELLBEING_SCORE = auto()


class ItemsToRecommendEnum(Enum):
    """The type of items to recommend."""

    GROUP = "group_based_activities"
    INDIVIDUAL = "individual_wellbeing_interventions"
    PHYSICAL = "pysical_wellbeing"
    SOCIAL = "social_support"
    TEAM = "team_cohesion"
    WORK = "work_design_and_flexibility"
    LEARNING = "learning_carrer_development"
    MEANING = "meaning_and_purpose"


class OnlineEventsEnum(StrEnum):
    """The type of online events possible."""

    RECOMMENDATION = auto()
    INTERACTION = auto()

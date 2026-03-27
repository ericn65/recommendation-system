from pydantic import BaseModel, Field

from recommendation_system.utils.enums import ItemsToRecommendEnum


class WellbeingDimensions(BaseModel):
    """Defining the dimensions to compute."""

    stress: float = Field(0.0, ge=0, le=1)
    burnout: float = Field(0.0, ge=0, le=1)
    motivation: float = Field(0.0, ge=0, le=1)
    resilience: float = Field(0.0, ge=0, le=1)
    autonomy: float = Field(0.0, ge=0, le=1)
    optimism: float = Field(0.0, ge=0, le=1)
    commitment: float = Field(0.0, ge=0, le=1)
    fulfilment: float = Field(0.0, ge=0, le=1)
    self_confidence: float = Field(0.0, ge=0, le=1)


class WellbeingSystemsFeatures(WellbeingDimensions):
    """All the needed wellbeing features."""

    general_wellbeing_score: float = Field(0.0, ge=0, le=1)


class ClientFeatures(BaseModel):
    """Different generical features."""

    sector: float = Field(0.0, ge=0, le=1)
    region: float = Field(0.0, ge=0, le=1)


class DemographicalAspectsFeatures(BaseModel):
    """The demographical aspects used as features."""

    # TODO: We should ensure how do we compute and use this stuff.
    # TODO: Create a model of how to receive it and functions to pass it.
    demographical_category_1: float = Field(0.0, ge=0, le=1)  # Must be the frequency.
    subdemographical_category_1: float = Field(0.0, ge=0, le=1)
    rate_of_participation: float = Field(0.0, ge=0, le=1)
    measurement: float = Field(0.0, ge=0, le=1)  # Must be a decimal number.
    demographical_category_2: float | None = None
    subdemographical_category_2: float | None = None
    actionable_tier: float | None = None  # Normalise some of the results somehow.
    intervention_done: bool = False


class UserFeatures(BaseModel):
    """Defining the different features a user array must have."""

    user_id: str
    wellbeing_features: WellbeingSystemsFeatures
    client_features: ClientFeatures
    demographical_aspects: DemographicalAspectsFeatures


class ItemsToRecommend(BaseModel):
    """All the possible items to reocmmend."""

    type: ItemsToRecommendEnum
    body: str


class ItemFeatures(BaseModel):
    """Defining the different features an item array must have."""

    item_id: ItemsToRecommend
    demographical_options: DemographicalAspectsFeatures
    client_options: ClientFeatures
    wellbeing_features: WellbeingSystemsFeatures

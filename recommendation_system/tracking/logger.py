import json
from datetime import datetime
from pathlib import Path

from recommendation_system.schemas.inference import ItemsToRecommend
from recommendation_system.utils.enums import OnlineEventsEnum
from recommendation_system.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationLogger:
    """
    Class to extract the feedback from clients on different recommendations.

    Parameters
    ----------
    path : str
        The path where this information will be hosted.
    """

    def __init__(self, log_path: str = "logs/recommendations.jsonl") -> None:
        """Initializes the recommendation Logger."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_recommendations(
        self,
        user_id: str,
        item_id: ItemsToRecommend,
        score: float,
        features: dict[str, float],
    ) -> None:
        """
        Logs the information in a JSON file to ensure we are getting all the
        specific informaiton.

        Parameters
        ----------
        user_id : str
            The user to keep track of.
        item_id : str
            The item recommended.
        score : float
            The confidence level for the recommendation.
        features : dict
            A list of the features given.
        """
        record = {
            "event": OnlineEventsEnum.RECOMMENDATION,
            "user_id": user_id,
            "item_id": item_id,
            "score": score,
            "features": features,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._write(record)

    def log_interaction(
        self, user_id: str, item_id: str, interation: dict[str, float]
    ) -> None:
        """
        Logs interactions done succesfully in the system to give feedback on
        which recommendations are being mostly used.

        Parameters
        ----------
        user_id : str
            The user to keep track of.
        item_id : str
            The item recommended.
        interactions : dict[str, float]
            The interactions happening with the user.
        """
        record = {
            "event": OnlineEventsEnum.INTERACTION,
            "user_id": user_id,
            "item_id": str(item_id),
            "interaction": interation,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._write(record)

    def _write(self, record: dict) -> None:
        """
        Writing the logs into the desired function to ensure there is a
        tracking and tracibility to compute the desired values.

        Parameters
        ----------
        record : dict[str, any]
            The records to be writing in the function.
        """
        try:
            serialized = self._serialize(record)

            with open(self.log_path, "a") as f:
                f.write(json.dumps(serialized) + "\n")

        except Exception as e:
            logger.exception(f"Error writing log: {e}")

    def _serialize(self, obj):
        """Recursively serialize objects for JSON dumping."""
        if hasattr(obj, "model_dump"):
            return self._serialize(obj.model_dump())

        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._serialize(v) for v in obj]

        if hasattr(obj, "value"):
            return obj.value

        return obj

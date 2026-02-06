from collections.abc import Iterable

import pandas as pd

from recommendation_system.metrics.precision import PrecisionAtK
from recommendation_system.pipelines.recommender_pipeline import RecommenderPipeline
from recommendation_system.recommenders.graph_recommender import GraphRecommender

model = GraphRecommender(alpha=0.9)

pipeline = RecommenderPipeline(model=model, metrics=[PrecisionAtK()])

interactions = pd.read_csv("READ-THE-PATH")
test_interactions = pd.read_csv("READ-THE-PATH")
test_users: Iterable = ...
pipeline.fit(interactions)  # TODO: Implement the aspects to get interactions.

scores = pipeline.evaluate(users=test_users, ground_truth=test_interactions, k=10)

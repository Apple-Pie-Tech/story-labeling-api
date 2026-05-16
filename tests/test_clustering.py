import pytest

from app.clustering import ClusteringError, HdbscanClusterer
from app.config import Settings
from app.vector_store import StoryPoint


def test_cluster_raises_when_too_few_points() -> None:
    clusterer = HdbscanClusterer(Settings(hdbscan_min_cluster_size=3))
    points = [
        StoryPoint(point_id="1", vector=[0.1, 0.2], text="one", payload={}),
        StoryPoint(point_id="2", vector=[0.2, 0.3], text="two", payload={}),
    ]

    with pytest.raises(ClusteringError, match="not enough points"):
        clusterer.cluster(points)


def test_cluster_raises_on_mixed_dimensions() -> None:
    clusterer = HdbscanClusterer(Settings(hdbscan_min_cluster_size=2))
    points = [
        StoryPoint(point_id="1", vector=[0.1, 0.2], text="one", payload={}),
        StoryPoint(point_id="2", vector=[0.2, 0.3, 0.4], text="two", payload={}),
    ]

    with pytest.raises(ClusteringError, match="mixed vector dimensions"):
        clusterer.cluster(points)

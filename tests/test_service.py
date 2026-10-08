import uuid
from dataclasses import dataclass

import pytest

from app.clustering import ClusterAssignments
from app.config import Settings
from app.labeling import ClusterTheme
from app.service import ClusterLabelingService
from app.vector_store import CentroidPoint, LoadedPoints, StoryPoint


class FakeStore:
    def __init__(self, points: list[StoryPoint]) -> None:
        self.points = points
        self.updates: dict[str | int, dict[str, object]] | None = None
        self.centroids: list[CentroidPoint] | None = None

    async def load_points(self) -> LoadedPoints:
        return LoadedPoints(points_read=len(self.points), valid_points=self.points)

    async def save_clustering_payloads(
        self,
        updates: dict[str | int, dict[str, object]],
    ) -> int:
        self.updates = updates
        return len(updates)

    async def replace_centroid_points(self, centroids: list[CentroidPoint]) -> int:
        self.centroids = centroids
        return len(centroids)

    async def aclose(self) -> None:
        return None


@dataclass
class FakeClusterer:
    labels: list[int]

    def cluster(self, points: list[StoryPoint]) -> ClusterAssignments:
        return ClusterAssignments(
            labels=self.labels,
            clusters_found=len({label for label in self.labels if label != -1}),
            noise_points=sum(1 for label in self.labels if label == -1),
        )


class FakeLabeler:
    async def label_cluster(self, cluster_id: int, points: list[StoryPoint]) -> ClusterTheme:
        if cluster_id == 1:
            raise RuntimeError("provider unavailable")
        return ClusterTheme(theme="Kitchen Stories", description="Memories around food.")


def make_points() -> list[StoryPoint]:
    return [
        StoryPoint(point_id="a", vector=[0.1, 0.2], text="apple", payload={}),
        StoryPoint(point_id="b", vector=[0.2, 0.3], text="pie", payload={}),
        StoryPoint(point_id="c", vector=[9.0, 9.1], text="science", payload={}),
        StoryPoint(point_id="d", vector=[7.0, 7.1], text="outlier", payload={}),
    ]


@pytest.mark.asyncio
async def test_service_writes_cluster_theme_and_noise_payloads() -> None:
    store = FakeStore(make_points())
    service = ClusterLabelingService(
        Settings(),
        store=store,
        clusterer=FakeClusterer(labels=[0, 0, 1, -1]),
        labeler=FakeLabeler(),
    )

    result = await service.run()

    assert result.status == "completed"
    assert result.points_read == 4
    assert result.points_clustered == 4
    assert result.clusters_found == 2
    assert result.noise_points == 1
    assert result.points_updated == 6

    assert store.updates is not None
    assert store.updates["a"] == {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": 0,
        "theme": "Kitchen Stories",
        "description": "Memories around food.",
        "is_noise": False,
    }
    assert store.updates["c"] == {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": 1,
        "theme": "Cluster 1",
        "description": None,
        "is_noise": False,
    }
    assert store.updates["d"] == {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": -1,
        "theme": "Noise / Outliers",
        "description": None,
        "is_noise": True,
    }

    # is_centroid stays top-level: data-provision-api's `_extract_label` reads
    # it there (never inside `clustering`) to set `is_central`, and this
    # service's own `_story_point_from_record` filters on the same top-level
    # key to keep centroids out of the next clustering run. Every other key
    # converges under `clustering`, matching the shape ordinary points get.
    assert store.centroids == [
        CentroidPoint(
            point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "centroid:hdbscan:0")),
            vector=[0.15000000596046448, 0.25],
            payload={
                "is_centroid": True,
                "clustering": {
                    "algorithm": "hdbscan",
                    "scope": "full_collection_original_embedding_space",
                    "centroid_key": "centroid:hdbscan:0",
                    "cluster_id": 0,
                    "theme": "Kitchen Stories",
                    "description": "Memories around food.",
                    "is_noise": False,
                },
            },
        ),
        CentroidPoint(
            point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "centroid:hdbscan:1")),
            vector=[9.0, 9.100000381469727],
            payload={
                "is_centroid": True,
                "clustering": {
                    "algorithm": "hdbscan",
                    "scope": "full_collection_original_embedding_space",
                    "centroid_key": "centroid:hdbscan:1",
                    "cluster_id": 1,
                    "theme": "Cluster 1",
                    "description": None,
                    "is_noise": False,
                },
            },
        ),
    ]


def test_centroid_point_id_is_a_valid_qdrant_point_id() -> None:
    """Qdrant only accepts an unsigned integer or a UUID as a point ID.

    The previous ``f"centroid:hdbscan:{cluster_id}"`` form is neither, so every
    real ``replace_centroid_points`` call failed with HTTP 400 and the whole
    labeling run died. The in-memory fakes accept any string, which is why the
    suite stayed green while the service was broken against a live Qdrant.
    """
    import uuid as _uuid

    from app.service import _centroid_point

    point = _centroid_point(
        cluster_id=0,
        vector=[0.0, 1.0],
        theme=ClusterTheme(theme="Morning Rides", description=None),
    )

    # Raises ValueError if not a well-formed UUID.
    _uuid.UUID(point.point_id)

    # Stable across calls, and distinct per cluster.
    again = _centroid_point(0, [0.0, 1.0], ClusterTheme(theme="Other", description=None))
    other = _centroid_point(1, [0.0, 1.0], ClusterTheme(theme="Morning Rides", description=None))
    assert point.point_id == again.point_id
    assert point.point_id != other.point_id

    # The human-readable key stays available for tracing.
    assert point.payload["clustering"]["centroid_key"] == "centroid:hdbscan:0"

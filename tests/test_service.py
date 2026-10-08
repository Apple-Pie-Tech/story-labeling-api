import uuid
from dataclasses import dataclass

import pytest

from app.clustering import ClusterAssignments
from app.config import Settings
from app.labeling import ClusterTheme
from app.service import ClusterLabelingService
from app.vector_store import (
    CentroidPoint,
    LoadedPoints,
    PointClusteringUpdate,
    StoryPoint,
)


class FakeStore:
    def __init__(
        self,
        points: list[StoryPoint],
        *,
        centroid_keys: tuple[str, ...] = (),
    ) -> None:
        self.points = points
        self.centroid_keys = centroid_keys
        self.updates: list[PointClusteringUpdate] | None = None
        self.centroids: list[CentroidPoint] | None = None
        self.existing_centroid_keys: tuple[str, ...] | None = None

    async def load_points(self) -> LoadedPoints:
        return LoadedPoints(
            points_read=len(self.points),
            valid_points=self.points,
            centroid_keys=self.centroid_keys,
        )

    async def save_clustering_payloads(
        self,
        updates: list[PointClusteringUpdate],
    ) -> int:
        self.updates = list(updates)
        return len(updates)

    async def replace_centroid_points(
        self,
        centroids: list[CentroidPoint],
        *,
        existing_centroid_keys: tuple[str, ...] = (),
    ) -> int:
        self.centroids = centroids
        self.existing_centroid_keys = tuple(existing_centroid_keys)
        return len(centroids)

    async def aclose(self) -> None:
        return None


def clustering_by_point(store: FakeStore) -> dict[str, dict[str, object]]:
    assert store.updates is not None
    return {update.point.point_id: update.clustering for update in store.updates}


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

    clustering = clustering_by_point(store)
    assert clustering["a"] == {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": 0,
        "theme": "Kitchen Stories",
        "description": "Memories around food.",
        "is_noise": False,
    }
    assert clustering["c"] == {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": 1,
        "theme": "Cluster 1",
        "description": None,
        "is_noise": False,
    }
    assert clustering["d"] == {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": -1,
        "theme": "Noise / Outliers",
        "description": None,
        "is_noise": True,
    }

    # The clustering result is written as flat `clustering_*` keys, because S3
    # Vectors rejects a nested metadata object outright. `is_centroid` stays
    # unprefixed: data-provision-api's `_extract_label` reads it there to set
    # `is_central`, and this service's read pass uses it both to keep centroids
    # out of the next clustering run and to find stale ones to delete.
    # A null description is dropped rather than stored -- an absent key and a
    # null mean the same thing to every reader.
    assert store.centroids == [
        CentroidPoint(
            point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "centroid:hdbscan:0")),
            vector=[0.15000000596046448, 0.25],
            payload={
                "is_centroid": True,
                "clustering_algorithm": "hdbscan",
                "clustering_scope": "full_collection_original_embedding_space",
                "clustering_centroid_key": "centroid:hdbscan:0",
                "clustering_cluster_id": 0,
                "clustering_theme": "Kitchen Stories",
                "clustering_description": "Memories around food.",
                "clustering_is_noise": False,
            },
        ),
        CentroidPoint(
            point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "centroid:hdbscan:1")),
            vector=[9.0, 9.100000381469727],
            payload={
                "is_centroid": True,
                "clustering_algorithm": "hdbscan",
                "clustering_scope": "full_collection_original_embedding_space",
                "clustering_centroid_key": "centroid:hdbscan:1",
                "clustering_cluster_id": 1,
                "clustering_theme": "Cluster 1",
                "clustering_is_noise": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_the_centroid_keys_from_the_read_pass_reach_the_write() -> None:
    """DeleteVectors has no filter, so the store needs the keys the read saw.

    Without this hand-off a shrinking corpus keeps centroids for clusters that
    no longer exist, and /universe shows phantom central points.
    """
    store = FakeStore(make_points(), centroid_keys=("centroid-key-0", "centroid-key-9"))
    service = ClusterLabelingService(
        Settings(),
        store=store,
        clusterer=FakeClusterer(labels=[0, 0, 1, -1]),
        labeler=FakeLabeler(),
    )

    await service.run()

    assert store.existing_centroid_keys == ("centroid-key-0", "centroid-key-9")


def test_centroid_point_id_is_stable_and_unique_per_cluster() -> None:
    """The centroid key is the identity contract across runs.

    S3 Vectors accepts any string as a key, so the UUID5 is no longer forced by
    the backend as it was by Qdrant (E49). It is kept because the same cluster
    must land on the same key every run: otherwise each run adds a new centroid
    instead of replacing the previous one.
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
    assert point.payload["clustering_centroid_key"] == "centroid:hdbscan:0"

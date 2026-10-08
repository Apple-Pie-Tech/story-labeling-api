from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.clustering import ClusterAssignments, HdbscanClusterer
from app.config import Settings
from app.labeling import BedrockClusterLabeler, ClusterTheme
from app.schemas import ClusterLabelResult
from app.vector_store import (
    CentroidPoint,
    LoadedPoints,
    PointClusteringUpdate,
    S3VectorsStoryStore,
    StoryPoint,
    clustering_metadata,
)


class Clusterer(Protocol):
    def cluster(self, points: list[StoryPoint]) -> ClusterAssignments: ...


class Labeler(Protocol):
    async def label_cluster(self, cluster_id: int, points: list[StoryPoint]) -> ClusterTheme: ...


class StoryStore(Protocol):
    async def load_points(self) -> LoadedPoints: ...

    async def save_clustering_payloads(
        self,
        updates: Sequence[PointClusteringUpdate],
    ) -> int: ...

    async def replace_centroid_points(
        self,
        centroids: Sequence[CentroidPoint],
        *,
        existing_centroid_keys: Sequence[str] = (),
    ) -> int: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class ClusterWriteSet:
    point_updates: list[PointClusteringUpdate]
    centroid_points: list[CentroidPoint]


class ClusterLabelingService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: StoryStore | None = None,
        clusterer: Clusterer | None = None,
        labeler: Labeler | None = None,
    ) -> None:
        self._store = store or S3VectorsStoryStore(settings)
        self._clusterer = clusterer or HdbscanClusterer(settings)
        self._labeler = labeler or BedrockClusterLabeler(settings)

    async def aclose(self) -> None:
        close_store = getattr(self._store, "aclose", None)
        if close_store is not None:
            await close_store()

        close_labeler = getattr(self._labeler, "aclose", None)
        if close_labeler is not None:
            await close_labeler()

    async def run(self) -> ClusterLabelResult:
        loaded = await self._store.load_points()
        assignments = self._clusterer.cluster(loaded.valid_points)
        write_set = await self._build_write_set(loaded.valid_points, assignments.labels)
        points_updated = await self._store.save_clustering_payloads(write_set.point_updates)
        centroids_updated = await self._store.replace_centroid_points(
            write_set.centroid_points,
            existing_centroid_keys=loaded.centroid_keys,
        )

        return ClusterLabelResult(
            status="completed",
            points_read=loaded.points_read,
            points_clustered=len(loaded.valid_points),
            clusters_found=assignments.clusters_found,
            noise_points=assignments.noise_points,
            points_updated=points_updated + centroids_updated,
        )

    async def _build_write_set(
        self,
        points: list[StoryPoint],
        labels: list[int],
    ) -> ClusterWriteSet:
        if len(points) != len(labels):
            raise RuntimeError("point and cluster label count mismatch")

        points_by_cluster: dict[int, list[StoryPoint]] = defaultdict(list)
        for point, label in zip(points, labels, strict=True):
            if label != -1:
                points_by_cluster[label].append(point)

        themes: dict[int, ClusterTheme] = {}
        for cluster_id, cluster_points in sorted(points_by_cluster.items()):
            try:
                themes[cluster_id] = await self._labeler.label_cluster(cluster_id, cluster_points)
            except Exception:
                themes[cluster_id] = ClusterTheme(
                    theme=f"Cluster {cluster_id}",
                    description=None,
                )

        centroids = {
            cluster_id: _median_centroid(cluster_points)
            for cluster_id, cluster_points in points_by_cluster.items()
        }

        updates: list[PointClusteringUpdate] = []
        for point, label in zip(points, labels, strict=True):
            if label == -1:
                updates.append(
                    PointClusteringUpdate(point=point, clustering=_noise_payload())
                )
                continue

            theme = themes[label]
            updates.append(
                PointClusteringUpdate(
                    point=point,
                    clustering={
                        "algorithm": "hdbscan",
                        "scope": "full_collection_original_embedding_space",
                        "cluster_id": label,
                        "theme": theme.theme,
                        "description": theme.description,
                        "is_noise": False,
                    },
                )
            )

        centroid_points = [
            _centroid_point(cluster_id, centroids[cluster_id], themes[cluster_id])
            for cluster_id in sorted(points_by_cluster)
        ]

        return ClusterWriteSet(point_updates=updates, centroid_points=centroid_points)


def _noise_payload() -> dict[str, object]:
    return {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": -1,
        "theme": "Noise / Outliers",
        "description": None,
        "is_noise": True,
    }


def _median_centroid(points: list[StoryPoint]) -> list[float]:
    matrix = np.array([point.vector for point in points], dtype=np.float32)
    return [float(value) for value in np.median(matrix, axis=0)]


def _centroid_point(
    cluster_id: int,
    vector: list[float],
    theme: ClusterTheme,
) -> CentroidPoint:
    # S3 Vectors keys are plain strings, so the UUID5 is no longer forced by the
    # backend as it was by Qdrant (E49). It is kept because it is the identity
    # contract: the same cluster must land on the same key across runs so a
    # centroid is replaced rather than duplicated. The readable key stays in the
    # payload.
    centroid_key = f"centroid:hdbscan:{cluster_id}"
    return CentroidPoint(
        point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, centroid_key)),
        vector=vector,
        payload={
            # Unprefixed, unlike every other clustering key: this is what
            # data-provision-api's reader checks to mark a point as central, and
            # what the next run's read pass uses to keep centroids out of
            # clustering and to find stale ones to delete.
            "is_centroid": True,
            **clustering_metadata(
                {
                    "algorithm": "hdbscan",
                    "scope": "full_collection_original_embedding_space",
                    "centroid_key": centroid_key,
                    "cluster_id": cluster_id,
                    "theme": theme.theme,
                    "description": theme.description,
                    "is_noise": False,
                }
            ),
        },
    )

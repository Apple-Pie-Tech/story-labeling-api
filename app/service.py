from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from app.clustering import ClusterAssignments, HdbscanClusterer
from app.config import Settings
from app.labeling import AzureClusterLabeler, ClusterTheme
from app.schemas import ClusterLabelResult
from app.vector_store import LoadedPoints, QdrantStoryStore, StoryPoint


class Clusterer(Protocol):
    def cluster(self, points: list[StoryPoint]) -> ClusterAssignments: ...


class Labeler(Protocol):
    async def label_cluster(self, cluster_id: int, points: list[StoryPoint]) -> ClusterTheme: ...


class StoryStore(Protocol):
    async def load_points(self) -> LoadedPoints: ...

    async def save_clustering_payloads(
        self,
        updates: dict[str | int, dict[str, object]],
    ) -> int: ...

    async def aclose(self) -> None: ...


class ClusterLabelingService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: StoryStore | None = None,
        clusterer: Clusterer | None = None,
        labeler: Labeler | None = None,
    ) -> None:
        self._store = store or QdrantStoryStore(settings)
        self._clusterer = clusterer or HdbscanClusterer(settings)
        self._labeler = labeler or AzureClusterLabeler(settings)

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
        updates = await self._build_updates(loaded.valid_points, assignments.labels)
        points_updated = await self._store.save_clustering_payloads(updates)

        return ClusterLabelResult(
            status="completed",
            points_read=loaded.points_read,
            points_clustered=len(loaded.valid_points),
            clusters_found=assignments.clusters_found,
            noise_points=assignments.noise_points,
            points_updated=points_updated,
        )

    async def _build_updates(
        self,
        points: list[StoryPoint],
        labels: list[int],
    ) -> dict[str | int, dict[str, object]]:
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

        updates: dict[str | int, dict[str, object]] = {}
        for point, label in zip(points, labels, strict=True):
            if label == -1:
                updates[point.point_id] = _noise_payload()
                continue

            theme = themes[label]
            updates[point.point_id] = {
                "algorithm": "hdbscan",
                "scope": "full_collection_original_embedding_space",
                "cluster_id": label,
                "theme": theme.theme,
                "description": theme.description,
                "is_noise": False,
            }

        return updates


def _noise_payload() -> dict[str, object]:
    return {
        "algorithm": "hdbscan",
        "scope": "full_collection_original_embedding_space",
        "cluster_id": -1,
        "theme": "Noise / Outliers",
        "description": None,
        "is_noise": True,
    }


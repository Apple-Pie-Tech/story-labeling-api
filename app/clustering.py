from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.vector_store import StoryPoint


class ClusteringError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClusterAssignments:
    labels: list[int]
    clusters_found: int
    noise_points: int


class HdbscanClusterer:
    def __init__(self, settings: Settings) -> None:
        self._min_cluster_size = settings.hdbscan_min_cluster_size
        self._min_samples = settings.hdbscan_min_samples

    def cluster(self, points: list[StoryPoint]) -> ClusterAssignments:
        if len(points) < self._min_cluster_size:
            raise ClusteringError(
                "not enough points to cluster: "
                f"need at least {self._min_cluster_size}, got {len(points)}"
            )

        dimensions = {len(point.vector) for point in points}
        if len(dimensions) != 1:
            raise ClusteringError(f"mixed vector dimensions found: {sorted(dimensions)}")

        matrix = np.array([point.vector for point in points], dtype=np.float32)

        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            metric="euclidean",
        )
        labels = [int(label) for label in clusterer.fit_predict(matrix)]
        clusters_found = len({label for label in labels if label != -1})
        noise_points = sum(1 for label in labels if label == -1)

        return ClusterAssignments(
            labels=labels,
            clusters_found=clusters_found,
            noise_points=noise_points,
        )

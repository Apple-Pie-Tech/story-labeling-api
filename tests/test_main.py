from fastapi.testclient import TestClient

from app.clustering import ClusteringError
from app.labeling import LabelingError
from app.main import app, get_cluster_labeling_service
from app.schemas import ClusterLabelResult
from app.vector_store import VectorStoreError


class FakeClusterLabelingService:
    async def run(self) -> ClusterLabelResult:
        return ClusterLabelResult(
            status="completed",
            points_read=5,
            points_clustered=4,
            clusters_found=2,
            noise_points=1,
            points_updated=6,
        )


class FailingClusteringService:
    async def run(self) -> ClusterLabelResult:
        raise ClusteringError("not enough points")


class FailingLabelingService:
    async def run(self) -> ClusterLabelResult:
        raise LabelingError("provider unavailable")


class FailingVectorStoreService:
    async def run(self) -> ClusterLabelResult:
        raise VectorStoreError("qdrant unavailable")


class FailingUnexpectedService:
    async def run(self) -> ClusterLabelResult:
        raise RuntimeError("boom")


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cluster_labels_returns_expected_contract() -> None:
    app.dependency_overrides[get_cluster_labeling_service] = lambda: FakeClusterLabelingService()
    client = TestClient(app)

    try:
        response = client.post("/cluster-labels")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "points_read": 5,
        "points_clustered": 4,
        "clusters_found": 2,
        "noise_points": 1,
        "points_updated": 6,
    }


def test_cluster_labels_maps_clustering_errors_to_400() -> None:
    app.dependency_overrides[get_cluster_labeling_service] = lambda: FailingClusteringService()
    client = TestClient(app)

    try:
        response = client.post("/cluster-labels")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {"detail": "not enough points"}


def test_cluster_labels_maps_labeling_errors_to_502() -> None:
    app.dependency_overrides[get_cluster_labeling_service] = lambda: FailingLabelingService()
    client = TestClient(app)

    try:
        response = client.post("/cluster-labels")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": "labeling_unavailable"}


def test_cluster_labels_maps_vector_store_errors_to_503() -> None:
    app.dependency_overrides[get_cluster_labeling_service] = lambda: FailingVectorStoreService()
    client = TestClient(app)

    try:
        response = client.post("/cluster-labels")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "vector_store_unavailable"}


def test_cluster_labels_maps_unexpected_errors_to_500() -> None:
    app.dependency_overrides[get_cluster_labeling_service] = lambda: FailingUnexpectedService()
    client = TestClient(app)

    try:
        response = client.post("/cluster-labels")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "internal_server_error"}

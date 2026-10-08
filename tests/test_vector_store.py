from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings
from app.vector_store import (
    MAX_VECTORS_PER_PUT,
    NON_FILTERABLE_METADATA_KEYS,
    CentroidPoint,
    PointClusteringUpdate,
    S3VectorsStoryStore,
    StoryPoint,
    VectorStoreConfigurationError,
    VectorStoreError,
    clustering_metadata,
    strip_clustering_metadata,
)


class FakeS3Vectors:
    """Stands in for boto3's s3vectors client.

    Paginates list_vectors over `pages`, exactly as the live service does with
    nextToken, so the read loop is exercised rather than assumed.
    """

    def __init__(
        self,
        pages: list[list[dict[str, Any]]] | None = None,
        *,
        list_error: Exception | None = None,
        put_error: Exception | None = None,
    ) -> None:
        self.pages = pages if pages is not None else [[]]
        self.list_error = list_error
        self.put_error = put_error
        self.list_calls: list[dict[str, Any]] = []
        self.puts: list[list[dict[str, Any]]] = []
        self.deletes: list[list[str]] = []
        self.closed = 0

    def list_vectors(self, **kwargs: Any) -> dict[str, Any]:
        self.list_calls.append(kwargs)
        if self.list_error is not None:
            raise self.list_error
        index = int(kwargs.get("nextToken", "0"))
        page = {"vectors": self.pages[index]}
        if index + 1 < len(self.pages):
            page["nextToken"] = str(index + 1)
        return page

    def put_vectors(self, **kwargs: Any) -> dict[str, Any]:
        if self.put_error is not None:
            raise self.put_error
        self.puts.append(kwargs["vectors"])
        return {}

    def delete_vectors(self, **kwargs: Any) -> dict[str, Any]:
        self.deletes.append(kwargs["keys"])
        return {}

    def close(self) -> None:
        self.closed += 1

    @property
    def put_vectors_flat(self) -> list[dict[str, Any]]:
        return [vector for batch in self.puts for vector in batch]


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "s3_vector_bucket": "applepie-vectors",
        "s3_vector_index": "apple-pie-story-chunks",
        "aws_region": "us-east-1",
        "vector_list_batch_size": 2,
        "vector_put_batch_size": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def record(
    key: str,
    vector: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"key": key}
    if vector is not None:
        out["data"] = {"float32": vector}
    if metadata is not None:
        out["metadata"] = metadata
    return out


def story_point(key: str = "input-1:0", **payload: Any) -> StoryPoint:
    metadata = {"text": "A story about apples.", "input_id": "input-1"}
    metadata.update(payload)
    return StoryPoint(
        point_id=key,
        vector=[0.1, 0.2],
        text="A story about apples.",
        payload=metadata,
    )


def test_missing_bucket_is_a_configuration_error() -> None:
    with pytest.raises(VectorStoreConfigurationError, match="S3_VECTOR_BUCKET"):
        S3VectorsStoryStore(make_settings(s3_vector_bucket=None), client=FakeS3Vectors())


@pytest.mark.asyncio
async def test_load_points_paginates_and_skips_unusable_records() -> None:
    """S3 Vectors has no scroll, so a full read is ListVectors to exhaustion."""
    client = FakeS3Vectors(
        [
            [
                record("input-1:0", [0.1, 0.2], {"text": "A story about apples."}),
                record("input-1:1", None, {"text": "Missing vector."}),
            ],
            [
                record("input-1:2", [0.3, 0.4], {"text": "A story about pie."}),
                record("input-1:3", [0.5, 0.6], {"not_text": "Missing text."}),
                record("centroid-key-0", [0.2, 0.3], {"is_centroid": True}),
            ],
        ]
    )
    store = S3VectorsStoryStore(make_settings(), client=client)

    loaded = await store.load_points()

    assert loaded.points_read == 5
    assert [point.point_id for point in loaded.valid_points] == ["input-1:0", "input-1:2"]
    assert [point.text for point in loaded.valid_points] == [
        "A story about apples.",
        "A story about pie.",
    ]
    assert len(client.list_calls) == 2
    assert all(call["returnData"] is True for call in client.list_calls)
    assert all(call["returnMetadata"] is True for call in client.list_calls)


@pytest.mark.asyncio
async def test_load_points_collects_existing_centroid_keys() -> None:
    """DeleteVectors has no filter, so stale centroids must be found while reading."""
    client = FakeS3Vectors(
        [
            [
                record("centroid-key-0", [0.2, 0.3], {"is_centroid": True}),
                record("centroid-key-1", [0.4, 0.5], {"is_centroid": True}),
                record("input-1:0", [0.1, 0.2], {"text": "A story."}),
            ]
        ]
    )
    store = S3VectorsStoryStore(make_settings(), client=client)

    loaded = await store.load_points()

    assert loaded.centroid_keys == ("centroid-key-0", "centroid-key-1")
    assert [point.point_id for point in loaded.valid_points] == ["input-1:0"]


@pytest.mark.asyncio
async def test_load_points_wraps_transport_failures() -> None:
    client = FakeS3Vectors(list_error=RuntimeError("AccessDeniedException"))
    store = S3VectorsStoryStore(make_settings(), client=client)

    with pytest.raises(VectorStoreError, match="list_vectors failed"):
        await store.load_points()


@pytest.mark.asyncio
async def test_save_clustering_payloads_resends_vector_and_existing_metadata() -> None:
    """PutVectors replaces metadata wholesale; there is no partial update.

    Proven live. So a label write has to carry the vector and the metadata that
    must survive, or the chunk loses its text and the index loses its vector.
    """
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    updated = await store.save_clustering_payloads(
        [
            PointClusteringUpdate(
                point=story_point(),
                clustering={
                    "algorithm": "hdbscan",
                    "cluster_id": 0,
                    "theme": "Apple memories",
                    "description": None,
                },
            )
        ]
    )

    assert updated == 1
    written = client.put_vectors_flat[0]
    assert written["key"] == "input-1:0"
    assert written["data"] == {"float32": [0.1, 0.2]}
    assert written["metadata"] == {
        "text": "A story about apples.",
        "input_id": "input-1",
        "clustering_algorithm": "hdbscan",
        "clustering_cluster_id": 0,
        "clustering_theme": "Apple memories",
    }


@pytest.mark.asyncio
async def test_a_relabelled_point_does_not_keep_the_previous_runs_keys() -> None:
    """A re-put must not leave a stale key describing a different cluster.

    The previous run wrote a description; this one has none. Merging without
    stripping would leave the old description attached to the new theme.
    """
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)
    stale = story_point(
        clustering_cluster_id=7,
        clustering_theme="Old theme",
        clustering_description="Stale description from the last run.",
        clustering_is_noise=False,
    )

    await store.save_clustering_payloads(
        [
            PointClusteringUpdate(
                point=stale,
                clustering={"cluster_id": 1, "theme": "New theme", "description": None},
            )
        ]
    )

    metadata = client.put_vectors_flat[0]["metadata"]
    assert metadata["clustering_cluster_id"] == 1
    assert metadata["clustering_theme"] == "New theme"
    assert "clustering_description" not in metadata
    assert "clustering_is_noise" not in metadata
    # Non-clustering metadata is untouched.
    assert metadata["text"] == "A story about apples."
    assert metadata["input_id"] == "input-1"


@pytest.mark.asyncio
async def test_no_metadata_value_is_a_nested_object() -> None:
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    await store.save_clustering_payloads(
        [
            PointClusteringUpdate(
                point=story_point(),
                clustering={"cluster_id": 0, "theme": "Apple memories", "is_noise": False},
            )
        ]
    )

    for key, value in client.put_vectors_flat[0]["metadata"].items():
        assert isinstance(value, str | int | float | bool | list), f"{key} is {type(value)}"


@pytest.mark.asyncio
async def test_a_nested_payload_is_rejected_with_a_readable_error() -> None:
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    with pytest.raises(VectorStoreError, match="nested object"):
        await store.replace_centroid_points(
            [
                CentroidPoint(
                    point_id="centroid-key-0",
                    vector=[0.1, 0.2],
                    payload={"is_centroid": True, "clustering": {"cluster_id": 0}},
                )
            ]
        )


@pytest.mark.asyncio
async def test_oversized_filterable_metadata_names_the_limit() -> None:
    """A long LLM-written theme is enough to breach the 2048-byte cap."""
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    with pytest.raises(VectorStoreError, match="2048-byte"):
        await store.save_clustering_payloads(
            [
                PointClusteringUpdate(
                    point=story_point(),
                    clustering={"cluster_id": 0, "theme": "t" * 3000},
                )
            ]
        )


@pytest.mark.asyncio
async def test_a_long_description_does_not_breach_the_cap() -> None:
    """clustering_description is non-filterable precisely so this works."""
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    await store.save_clustering_payloads(
        [
            PointClusteringUpdate(
                point=story_point(),
                clustering={"cluster_id": 0, "theme": "Apples", "description": "d" * 5000},
            )
        ]
    )

    assert len(client.put_vectors_flat[0]["metadata"]["clustering_description"]) == 5000


@pytest.mark.asyncio
async def test_writes_are_split_across_put_vectors_calls() -> None:
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(vector_put_batch_size=2), client=client)

    updates = [
        PointClusteringUpdate(
            point=story_point(key=f"input-1:{index}"),
            clustering={"cluster_id": 0, "theme": "Apples"},
        )
        for index in range(5)
    ]

    updated = await store.save_clustering_payloads(updates)

    assert updated == 5
    assert [len(batch) for batch in client.puts] == [2, 2, 1]


def test_put_batch_size_cannot_exceed_the_api_limit() -> None:
    """PutVectors accepts at most 500 vectors; a larger setting must be clamped."""
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(vector_put_batch_size=5000), client=client)

    assert store._config.put_batch_size == MAX_VECTORS_PER_PUT


@pytest.mark.asyncio
async def test_replace_centroid_points_deletes_only_the_stale_ones() -> None:
    """Idempotent: a centroid being rewritten must not be deleted first.

    Qdrant deleted every centroid by filter and re-upserted. Doing that here
    would mean a window with no centroid for a cluster that still has one, and
    DeleteVectors has no filter to do it atomically anyway.
    """
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    updated = await store.replace_centroid_points(
        [
            CentroidPoint(
                point_id="centroid-key-0",
                vector=[0.15, 0.25],
                payload={"is_centroid": True, "clustering_cluster_id": 0},
            )
        ],
        existing_centroid_keys=("centroid-key-0", "centroid-key-1", "centroid-key-2"),
    )

    assert updated == 1
    assert client.deletes == [["centroid-key-1", "centroid-key-2"]]
    assert client.put_vectors_flat[0]["key"] == "centroid-key-0"
    assert client.put_vectors_flat[0]["data"] == {"float32": [0.15, 0.25]}


@pytest.mark.asyncio
async def test_replace_centroid_points_with_no_clusters_still_clears_stale_ones() -> None:
    """A corpus that no longer clusters must not keep last run's centroids."""
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    updated = await store.replace_centroid_points(
        [], existing_centroid_keys=("centroid-key-0",)
    )

    assert updated == 0
    assert client.deletes == [["centroid-key-0"]]
    assert client.puts == []


@pytest.mark.asyncio
async def test_no_delete_call_is_made_when_nothing_is_stale() -> None:
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    await store.replace_centroid_points(
        [
            CentroidPoint(
                point_id="centroid-key-0",
                vector=[0.1, 0.2],
                payload={"is_centroid": True},
            )
        ],
        existing_centroid_keys=("centroid-key-0",),
    )

    assert client.deletes == []


@pytest.mark.asyncio
async def test_put_failures_are_wrapped() -> None:
    client = FakeS3Vectors(put_error=RuntimeError("ThrottlingException"))
    store = S3VectorsStoryStore(make_settings(), client=client)

    with pytest.raises(VectorStoreError, match="put_vectors failed"):
        await store.save_clustering_payloads(
            [PointClusteringUpdate(point=story_point(), clustering={"cluster_id": 0})]
        )


@pytest.mark.asyncio
async def test_an_injected_client_is_not_closed_by_the_store() -> None:
    client = FakeS3Vectors()
    store = S3VectorsStoryStore(make_settings(), client=client)

    await store.aclose()

    assert client.closed == 0


def test_clustering_metadata_prefixes_and_drops_nulls() -> None:
    assert clustering_metadata({"cluster_id": 0, "theme": "A", "description": None}) == {
        "clustering_cluster_id": 0,
        "clustering_theme": "A",
    }


def test_strip_clustering_metadata_leaves_everything_else() -> None:
    assert strip_clustering_metadata(
        {"text": "t", "is_centroid": True, "clustering_theme": "A"}
    ) == {"text": "t", "is_centroid": True}


def test_text_and_description_are_declared_non_filterable() -> None:
    """Both can exceed the 2048-byte filterable cap on their own."""
    assert "text" in NON_FILTERABLE_METADATA_KEYS
    assert "clustering_description" in NON_FILTERABLE_METADATA_KEYS
    assert len(NON_FILTERABLE_METADATA_KEYS) <= 10

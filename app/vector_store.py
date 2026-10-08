from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

from app.config import Settings


# Verified against the live service rather than taken from the docs:
#
#   * metadata values must be strings, numbers, booleans or arrays -- a nested
#     object is rejected with ValidationException. The Qdrant payload nested the
#     clustering result under `clustering`, so those keys are flattened with the
#     prefix below. data-ingestion's vector_store.py holds the matching list for
#     the keys it writes; the two have to agree, and the index pins them.
#   * filterable metadata is capped at 2048 bytes per vector, so `text` and the
#     LLM-written `clustering_description` are declared non-filterable on the
#     index at creation time and cannot be made filterable later.
#   * PutVectors replaces a key's metadata WHOLESALE. There is no set_payload
#     equivalent, which is why writing a label back needs the point's vector and
#     its existing metadata, not just the new keys.
#   * DeleteVectors takes explicit keys only -- it has no filter -- so stale
#     centroids must be identified during the read pass.
MAX_VECTORS_PER_PUT = 500
MAX_VECTORS_PER_DELETE = 500
MAX_VECTORS_PER_LIST = 1000
MAX_FILTERABLE_METADATA_BYTES = 2048
NON_FILTERABLE_METADATA_KEYS: tuple[str, ...] = (
    "text",
    "audio_url",
    "clustering_description",
)
CLUSTERING_METADATA_PREFIX = "clustering_"


class VectorStoreError(RuntimeError):
    pass


class VectorStoreConfigurationError(VectorStoreError):
    pass


@dataclass(frozen=True)
class StoryPoint:
    point_id: str
    vector: list[float]
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class LoadedPoints:
    points_read: int
    valid_points: list[StoryPoint]
    # Centroid keys already in the index. DeleteVectors has no filter, so the
    # only cheap way to find stale centroids is to note them while reading.
    centroid_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class CentroidPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, object]


@dataclass(frozen=True)
class PointClusteringUpdate:
    """A story point together with the clustering result to record on it.

    Carries the whole point, not just its id, because PutVectors replaces
    metadata wholesale: re-sending the vector and the surviving metadata is the
    only way to add a key.
    """

    point: StoryPoint
    clustering: dict[str, object]


def clustering_metadata(clustering: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a clustering result into flat, prefixed metadata keys.

    Nulls are dropped rather than stored: an absent key and a null mean the same
    thing to every reader, and it keeps the payload within the value types S3
    Vectors accepts.
    """

    return {
        f"{CLUSTERING_METADATA_PREFIX}{key}": value
        for key, value in clustering.items()
        if value is not None
    }


def strip_clustering_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop every clustering key from a payload.

    Applied before a re-put so a key the previous run wrote but this one does
    not -- a description that came back empty, say -- does not survive as stale
    metadata attached to a different cluster.
    """

    return {
        key: value
        for key, value in payload.items()
        if not key.startswith(CLUSTERING_METADATA_PREFIX)
    }


@runtime_checkable
class S3VectorsAPI(Protocol):
    def list_vectors(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_vectors(self, **kwargs: Any) -> dict[str, Any]: ...

    def delete_vectors(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass
class _StoreConfig:
    bucket: str
    index: str
    list_batch_size: int
    put_batch_size: int


class S3VectorsStoryStore:
    """Reads the whole index, writes cluster labels back into it.

    boto3 is synchronous, so calls hop to a worker thread.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: S3VectorsAPI | None = None,
    ) -> None:
        if not settings.s3_vector_bucket:
            raise VectorStoreConfigurationError("S3_VECTOR_BUCKET is required")
        if not settings.s3_vector_index:
            raise VectorStoreConfigurationError("S3_VECTOR_INDEX is required")

        self._config = _StoreConfig(
            bucket=settings.s3_vector_bucket,
            index=settings.s3_vector_index,
            list_batch_size=min(settings.vector_list_batch_size, MAX_VECTORS_PER_LIST),
            put_batch_size=min(settings.vector_put_batch_size, MAX_VECTORS_PER_PUT),
        )
        self._owns_client = client is None
        if client is not None:
            self._client: Any = client
        else:
            boto3 = import_module("boto3")
            self._client = boto3.client("s3vectors", region_name=settings.aws_region)

    async def aclose(self) -> None:
        if not self._owns_client:
            return

        close = getattr(self._client, "close", None)
        if close is None:
            return
        await asyncio.to_thread(close)

    async def load_points(self) -> LoadedPoints:
        """Read every vector in the index, with its data and metadata.

        S3 Vectors has no Qdrant-style scroll, so a full read is ListVectors
        paginated to exhaustion. This is fine at the current corpus size and is
        the thing to revisit first if it grows past a few thousand points.
        """
        valid_points: list[StoryPoint] = []
        centroid_keys: list[str] = []
        points_read = 0
        next_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "vectorBucketName": self._config.bucket,
                "indexName": self._config.index,
                "returnData": True,
                "returnMetadata": True,
                "maxResults": self._config.list_batch_size,
            }
            if next_token:
                kwargs["nextToken"] = next_token

            try:
                page = await asyncio.to_thread(self._client.list_vectors, **kwargs)
            except Exception as exc:
                raise VectorStoreError(f"S3 Vectors list_vectors failed: {exc}") from exc

            records = page.get("vectors") or []
            points_read += len(records)
            for record in records:
                if _is_centroid(record):
                    key = record.get("key")
                    if isinstance(key, str):
                        centroid_keys.append(key)
                    continue

                story_point = _story_point_from_record(record)
                if story_point is not None:
                    valid_points.append(story_point)

            next_token = page.get("nextToken")
            if not next_token:
                break

        return LoadedPoints(
            points_read=points_read,
            valid_points=valid_points,
            centroid_keys=tuple(centroid_keys),
        )

    async def save_clustering_payloads(
        self,
        updates: Sequence[PointClusteringUpdate],
    ) -> int:
        vectors = [
            {
                "key": update.point.point_id,
                "data": {"float32": [float(value) for value in update.point.vector]},
                "metadata": _merged_metadata(update),
            }
            for update in updates
        ]
        await self._put(vectors)
        return len(vectors)

    async def replace_centroid_points(
        self,
        centroids: Sequence[CentroidPoint],
        *,
        existing_centroid_keys: Sequence[str] = (),
    ) -> int:
        """Write this run's centroids and delete any the last run left behind.

        Qdrant deleted centroids with a filter on `is_centroid`. DeleteVectors
        has no filter, so the keys come from the read pass; only centroids this
        run is not rewriting are deleted, which keeps the operation idempotent
        and never removes a centroid it is about to replace.
        """
        written_keys = {centroid.point_id for centroid in centroids}
        stale_keys = [key for key in existing_centroid_keys if key not in written_keys]
        await self._delete(stale_keys)

        if not centroids:
            return 0

        vectors = [
            {
                "key": centroid.point_id,
                "data": {"float32": [float(value) for value in centroid.vector]},
                "metadata": _validated_metadata(dict(centroid.payload), centroid.point_id),
            }
            for centroid in centroids
        ]
        await self._put(vectors)
        return len(vectors)

    async def _put(self, vectors: list[dict[str, Any]]) -> None:
        for batch in _batched(vectors, self._config.put_batch_size):
            try:
                await asyncio.to_thread(
                    self._client.put_vectors,
                    vectorBucketName=self._config.bucket,
                    indexName=self._config.index,
                    vectors=batch,
                )
            except Exception as exc:
                raise VectorStoreError(f"S3 Vectors put_vectors failed: {exc}") from exc

    async def _delete(self, keys: list[str]) -> None:
        for batch in _batched(keys, MAX_VECTORS_PER_DELETE):
            try:
                await asyncio.to_thread(
                    self._client.delete_vectors,
                    vectorBucketName=self._config.bucket,
                    indexName=self._config.index,
                    keys=batch,
                )
            except Exception as exc:
                raise VectorStoreError(f"S3 Vectors delete_vectors failed: {exc}") from exc


def _merged_metadata(update: PointClusteringUpdate) -> dict[str, Any]:
    merged = strip_clustering_metadata(update.point.payload)
    merged.update(clustering_metadata(update.clustering))
    return _validated_metadata(merged, update.point.point_id)


def _validated_metadata(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    """Reject metadata S3 Vectors would reject, with a message that says why.

    A long LLM-written theme is enough to breach the 2048-byte filterable cap,
    and the service's own 400 names neither the cap's cause nor the keys
    responsible.
    """
    filterable = {
        name: value
        for name, value in metadata.items()
        if name not in NON_FILTERABLE_METADATA_KEYS
    }
    for name, value in filterable.items():
        if isinstance(value, Mapping):
            raise VectorStoreError(
                f"metadata key {name!r} on {key} is a nested object, "
                "which S3 Vectors rejects"
            )

    size = len(json.dumps(filterable, separators=(",", ":"), default=str).encode("utf-8"))
    if size > MAX_FILTERABLE_METADATA_BYTES:
        raise VectorStoreError(
            f"filterable metadata for {key} is {size} bytes, over the "
            f"{MAX_FILTERABLE_METADATA_BYTES}-byte S3 Vectors limit; "
            f"keys: {sorted(filterable)}"
        )
    return metadata


def _batched(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def _is_centroid(record: Mapping[str, Any]) -> bool:
    metadata = record.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("is_centroid") is True


def _story_point_from_record(record: Mapping[str, Any]) -> StoryPoint | None:
    point_id = record.get("key")
    metadata = record.get("metadata")

    if not isinstance(point_id, str) or not point_id:
        return None
    if not isinstance(metadata, Mapping):
        return None

    data = record.get("data")
    if not isinstance(data, Mapping):
        return None

    vector = data.get("float32")
    if not isinstance(vector, list) or not vector:
        return None

    text = metadata.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        normalized_vector = [float(value) for value in vector]
    except (TypeError, ValueError):
        return None

    return StoryPoint(
        point_id=point_id,
        vector=normalized_vector,
        text=text.strip(),
        payload=dict(metadata),
    )

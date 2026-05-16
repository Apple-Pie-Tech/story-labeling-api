from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import Settings


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoryPoint:
    point_id: str | int
    vector: list[float]
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class LoadedPoints:
    points_read: int
    valid_points: list[StoryPoint]


@dataclass(frozen=True)
class CentroidPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, object]


@runtime_checkable
class QdrantPointsAPI(Protocol):
    async def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: object | None = None,
        with_payload: bool = True,
        with_vectors: bool = True,
        **kwargs: object,
    ) -> tuple[list[object], object | None]: ...

    async def set_payload(
        self,
        collection_name: str,
        *,
        payload: dict[str, object],
        points: list[str | int],
        wait: bool = True,
        **kwargs: object,
    ) -> object: ...

    async def delete(
        self,
        collection_name: str,
        *,
        points_selector: object,
        wait: bool = True,
        **kwargs: object,
    ) -> object: ...

    async def upsert(
        self,
        collection_name: str,
        *,
        points: list[qdrant_models.PointStruct],
        wait: bool = True,
        **kwargs: object,
    ) -> object: ...


class QdrantStoryStore:
    def __init__(
        self,
        settings: Settings,
        *,
        client: QdrantPointsAPI | None = None,
    ) -> None:
        self._collection_name = settings.qdrant_collection
        self._scroll_batch_size = settings.qdrant_scroll_batch_size
        self._update_batch_size = settings.qdrant_update_batch_size
        self._client: QdrantPointsAPI | AsyncQdrantClient = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if not self._owns_client:
            return

        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def load_points(self) -> LoadedPoints:
        valid_points: list[StoryPoint] = []
        points_read = 0
        offset: object | None = None

        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection_name,
                limit=self._scroll_batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points_read += len(records)
            for record in records:
                story_point = _story_point_from_record(record)
                if story_point is not None:
                    valid_points.append(story_point)

            if offset is None:
                break

        return LoadedPoints(points_read=points_read, valid_points=valid_points)

    async def save_clustering_payloads(
        self,
        updates: dict[str | int, dict[str, object]],
    ) -> int:
        updated = 0
        items = list(updates.items())
        for index in range(0, len(items), self._update_batch_size):
            batch = items[index : index + self._update_batch_size]
            for point_id, clustering_payload in batch:
                await self._client.set_payload(
                    collection_name=self._collection_name,
                    payload={"clustering": clustering_payload},
                    points=[point_id],
                    wait=True,
                )
                updated += 1

        return updated

    async def replace_centroid_points(self, centroids: list[CentroidPoint]) -> int:
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="is_centroid",
                            match=qdrant_models.MatchValue(value=True),
                        )
                    ]
                )
            ),
            wait=True,
        )

        if not centroids:
            return 0

        points = [
            qdrant_models.PointStruct(
                id=centroid.point_id,
                vector=centroid.vector,
                payload=centroid.payload,
            )
            for centroid in centroids
        ]
        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )
        return len(points)


def _story_point_from_record(record: object) -> StoryPoint | None:
    point_id = getattr(record, "id", None)
    vector = getattr(record, "vector", None)
    payload = getattr(record, "payload", None)

    if point_id is None or not isinstance(payload, dict):
        return None

    if payload.get("is_centroid") is True:
        return None

    if isinstance(vector, dict):
        return None

    if not isinstance(vector, list) or not vector:
        return None

    text = payload.get("text")
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
        payload=payload,
    )

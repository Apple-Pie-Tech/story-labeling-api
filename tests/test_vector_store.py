from types import SimpleNamespace

import pytest

from app.config import Settings
from app.vector_store import QdrantStoryStore


class FakeQdrantClient:
    def __init__(self) -> None:
        self.scroll_calls = 0
        self.payload_updates: list[dict[str, object]] = []
        self.deleted_selectors: list[object] = []
        self.upserted_points: list[object] = []

    async def scroll(self, **kwargs: object) -> tuple[list[object], object | None]:
        self.scroll_calls += 1
        if self.scroll_calls == 1:
            return (
                [
                    SimpleNamespace(
                        id="input-1:0",
                        vector=[0.1, 0.2],
                        payload={"text": "A story about apples."},
                    ),
                    SimpleNamespace(
                        id="input-1:1",
                        vector=None,
                        payload={"text": "Missing vector."},
                    ),
                ],
                "next",
            )

        return (
            [
                SimpleNamespace(
                    id="input-1:2",
                    vector=[0.3, 0.4],
                    payload={"text": "A story about pie."},
                ),
                    SimpleNamespace(
                        id="input-1:3",
                        vector=[0.5, 0.6],
                        payload={"not_text": "Missing text."},
                    ),
                    SimpleNamespace(
                        id="centroid:hdbscan:0",
                        vector=[0.2, 0.3],
                        payload={"is_centroid": True, "theme": "Old centroid"},
                    ),
                ],
                None,
            )

    async def set_payload(self, **kwargs: object) -> object:
        self.payload_updates.append(kwargs)
        return SimpleNamespace(status="acknowledged")

    async def delete(self, **kwargs: object) -> object:
        self.deleted_selectors.append(kwargs["points_selector"])
        return SimpleNamespace(status="acknowledged")

    async def upsert(self, **kwargs: object) -> object:
        self.upserted_points = kwargs["points"]
        return SimpleNamespace(status="acknowledged")


def make_settings() -> Settings:
    return Settings(
        qdrant_url="http://qdrant:6333",
        qdrant_collection="apple_pie_story_chunks",
        qdrant_scroll_batch_size=2,
        qdrant_update_batch_size=2,
    )


@pytest.mark.asyncio
async def test_load_points_scrolls_and_skips_unusable_records() -> None:
    client = FakeQdrantClient()
    store = QdrantStoryStore(make_settings(), client=client)

    loaded = await store.load_points()

    assert loaded.points_read == 5
    assert [point.point_id for point in loaded.valid_points] == ["input-1:0", "input-1:2"]
    assert [point.text for point in loaded.valid_points] == [
        "A story about apples.",
        "A story about pie.",
    ]


@pytest.mark.asyncio
async def test_save_clustering_payloads_preserves_payload_under_clustering_key() -> None:
    client = FakeQdrantClient()
    store = QdrantStoryStore(make_settings(), client=client)

    updated = await store.save_clustering_payloads(
        {
            "input-1:0": {
                "algorithm": "hdbscan",
                "cluster_id": 0,
                "theme": "Apple memories",
            }
        }
    )

    assert updated == 1
    assert client.payload_updates == [
        {
            "collection_name": "apple_pie_story_chunks",
            "payload": {
                "clustering": {
                    "algorithm": "hdbscan",
                    "cluster_id": 0,
                    "theme": "Apple memories",
                }
            },
            "points": ["input-1:0"],
            "wait": True,
        }
    ]


@pytest.mark.asyncio
async def test_replace_centroid_points_deletes_old_centroids_and_upserts_new_ones() -> None:
    client = FakeQdrantClient()
    store = QdrantStoryStore(make_settings(), client=client)

    updated = await store.replace_centroid_points(
        [
            SimpleNamespace(
                point_id="centroid:hdbscan:0",
                vector=[0.15, 0.25],
                payload={
                    "is_centroid": True,
                    "cluster_id": 0,
                    "theme": "Kitchen Stories",
                    "description": "Food memories.",
                },
            )
        ]
    )

    assert updated == 1
    assert len(client.deleted_selectors) == 1
    assert len(client.upserted_points) == 1
    assert client.upserted_points[0].id == "centroid:hdbscan:0"
    assert client.upserted_points[0].vector == [0.15, 0.25]
    assert client.upserted_points[0].payload == {
        "is_centroid": True,
        "cluster_id": 0,
        "theme": "Kitchen Stories",
        "description": "Food memories.",
    }

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.labeling import AzureClusterLabeler, LabelingError
from app.vector_store import StoryPoint


class FakeChatCompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


@pytest.mark.asyncio
async def test_label_cluster_parses_json_theme() -> None:
    api = FakeChatCompletions('{"theme": "Kitchen Stories", "description": "Food memories."}')
    labeler = AzureClusterLabeler(Settings(), chat_completions_api=api)
    points = [StoryPoint(point_id="1", vector=[0.1], text="Apple pie at home.", payload={})]

    theme = await labeler.label_cluster(0, points)

    assert theme.theme == "Kitchen Stories"
    assert theme.description == "Food memories."
    assert api.calls[0]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_label_cluster_rejects_non_json_response() -> None:
    labeler = AzureClusterLabeler(
        Settings(),
        chat_completions_api=FakeChatCompletions("not json"),
    )
    points = [StoryPoint(point_id="1", vector=[0.1], text="Apple pie at home.", payload={})]

    with pytest.raises(LabelingError):
        await labeler.label_cluster(0, points)

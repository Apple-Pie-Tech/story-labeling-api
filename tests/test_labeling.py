from types import SimpleNamespace

import pytest

from app.config import Settings
from app.labeling import BedrockClusterLabeler, BedrockConverseAPI, LabelingError
from app.vector_store import StoryPoint


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_points() -> list[StoryPoint]:
    return [StoryPoint(point_id="1", vector=[0.1], text="Apple pie at home.", payload={})]


class FakeChatCompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeBedrockRuntime:
    """Stands in for boto3's bedrock-runtime client on the Converse path."""

    def __init__(self, text: str = '{"theme": "Kitchen Stories"}', stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, object]] = []
        self._text = text
        self._stop_reason = stop_reason

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "stopReason": self._stop_reason,
            "output": {"message": {"content": [{"text": self._text}]}},
        }


@pytest.mark.asyncio
async def test_label_cluster_parses_json_theme() -> None:
    api = FakeChatCompletions('{"theme": "Kitchen Stories", "description": "Food memories."}')
    labeler = BedrockClusterLabeler(make_settings(), chat_completions_api=api)

    theme = await labeler.label_cluster(0, make_points())

    assert theme.theme == "Kitchen Stories"
    assert theme.description == "Food memories."
    assert api.calls[0]["model"] == "amazon.nova-pro-v1:0"
    assert api.calls[0]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_label_cluster_sends_configured_temperature() -> None:
    api = FakeChatCompletions('{"theme": "Kitchen Stories", "description": null}')
    labeler = BedrockClusterLabeler(
        make_settings(bedrock_labeling_temperature=0.7),
        chat_completions_api=api,
    )

    await labeler.label_cluster(0, make_points())

    assert api.calls[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_label_cluster_omits_temperature_when_unset() -> None:
    """Some model families reject any explicit temperature, including the default.

    The parameter must be left out of the call entirely, not merely set to the
    default value.
    """
    api = FakeChatCompletions('{"theme": "Kitchen Stories", "description": null}')
    labeler = BedrockClusterLabeler(
        make_settings(bedrock_labeling_temperature=None),
        chat_completions_api=api,
    )

    await labeler.label_cluster(0, make_points())

    assert "temperature" not in api.calls[0]


@pytest.mark.asyncio
async def test_label_cluster_rejects_non_json_response() -> None:
    labeler = BedrockClusterLabeler(
        make_settings(),
        chat_completions_api=FakeChatCompletions("not json"),
    )

    with pytest.raises(LabelingError):
        await labeler.label_cluster(0, make_points())


@pytest.mark.asyncio
async def test_fenced_json_is_tolerated() -> None:
    """Nova Pro returns bare JSON but Nova Lite and Micro wrap it in ```json fences.

    Switching model should not break labelling with "label response was not valid JSON".
    """
    api = FakeChatCompletions('```json\n{"theme": "Kitchen Stories", "description": null}\n```')
    labeler = BedrockClusterLabeler(make_settings(), chat_completions_api=api)

    theme = await labeler.label_cluster(0, make_points())

    assert theme.theme == "Kitchen Stories"


@pytest.mark.asyncio
async def test_converse_moves_system_prompt_out_of_messages() -> None:
    """Converse takes system prompts in their own argument, not as a message role.

    Leaving a system-role entry in `messages` makes Bedrock reject the request.
    """
    runtime = FakeBedrockRuntime()
    api = BedrockConverseAPI(region="us-east-1", max_tokens=512, client=runtime)

    await api.create(
        model="amazon.nova-pro-v1:0",
        messages=[
            {"role": "system", "content": "You label clusters."},
            {"role": "user", "content": "Cluster 0 contains..."},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    call = runtime.calls[0]
    assert [turn["role"] for turn in call["messages"]] == ["user"]
    assert call["messages"][0]["content"] == [{"text": "Cluster 0 contains..."}]
    assert call["system"][0] == {"text": "You label clusters."}
    assert call["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.1}


@pytest.mark.asyncio
async def test_converse_translates_json_object_into_a_prompt_instruction() -> None:
    """Nova has no native response_format, so JSON mode has to become an instruction."""
    runtime = FakeBedrockRuntime()
    api = BedrockConverseAPI(region="us-east-1", max_tokens=512, client=runtime)

    await api.create(
        model="amazon.nova-pro-v1:0",
        messages=[{"role": "user", "content": "label this"}],
        response_format={"type": "json_object"},
    )

    system_texts = [block["text"] for block in runtime.calls[0]["system"]]
    assert any("single JSON object" in text for text in system_texts)
    # No temperature was passed, so none should reach Bedrock.
    assert "temperature" not in runtime.calls[0]["inferenceConfig"]


@pytest.mark.asyncio
async def test_converse_reports_token_truncation_explicitly() -> None:
    """A truncated response is invalid JSON; the real cause must not be hidden."""
    runtime = FakeBedrockRuntime(text='{"theme": "Kitch', stop_reason="max_tokens")
    api = BedrockConverseAPI(region="us-east-1", max_tokens=8, client=runtime)

    with pytest.raises(LabelingError, match="BEDROCK_LABELING_MAX_TOKENS"):
        await api.create(
            model="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "label this"}],
            response_format={"type": "json_object"},
        )


@pytest.mark.asyncio
async def test_converse_transport_failure_is_wrapped() -> None:
    class ExplodingRuntime:
        def converse(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("throttled")

    api = BedrockConverseAPI(region="us-east-1", max_tokens=512, client=ExplodingRuntime())

    with pytest.raises(LabelingError, match="Bedrock Converse request failed"):
        await api.create(
            model="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )

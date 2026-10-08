from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, cast, runtime_checkable

from app.config import Settings
from app.vector_store import StoryPoint


class LabelingError(RuntimeError):
    pass


class LabelingConfigurationError(LabelingError):
    pass


@dataclass(frozen=True)
class ClusterTheme:
    theme: str
    description: str | None


@runtime_checkable
class ChatCompletionsAPI(Protocol):
    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, str],
        temperature: float = ...,
    ) -> object: ...


class _ChoiceMessage(Protocol):
    content: str | None


class _Choice(Protocol):
    message: _ChoiceMessage


class _ChatResponse(Protocol):
    choices: list[_Choice]


@dataclass(frozen=True)
class _Message:
    content: str | None


@dataclass(frozen=True)
class _ResponseChoice:
    message: _Message


@dataclass(frozen=True)
class _Response:
    choices: list[_ResponseChoice] = field(default_factory=list)


# Nova has no native JSON mode, unlike Azure OpenAI's response_format. The only
# lever is the prompt, so a json_object request is translated into this
# instruction. Verified against Nova Pro: it returns bare, parseable JSON.
_JSON_ONLY_INSTRUCTION = (
    "Respond with a single JSON object and nothing else. "
    "Do not wrap it in Markdown code fences and do not add commentary."
)


class BedrockConverseAPI:
    """Bedrock Converse behind the chat-completions interface this service used.

    Nova is not available on Bedrock's OpenAI-compatible Chat Completions path, only
    on Converse, so the request has to be reshaped: system prompts move out of
    `messages` into their own argument, content becomes a list of blocks, and
    sampling settings move into `inferenceConfig`. Presenting an OpenAI-shaped
    response back keeps `label_cluster` and every injected test double unchanged.

    boto3 is synchronous, so the call runs in a worker thread rather than blocking
    the event loop for the whole round-trip.
    """

    def __init__(
        self,
        *,
        region: str,
        max_tokens: int,
        client: Any | None = None,
    ) -> None:
        if client is None:
            boto3 = import_module("boto3")
            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self._max_tokens = max_tokens

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, str] | None = None,
        temperature: float | None = None,
    ) -> object:
        system_prompts = [m["content"] for m in messages if m.get("role") == "system"]
        if response_format and response_format.get("type") == "json_object":
            system_prompts.append(_JSON_ONLY_INSTRUCTION)

        turns = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
            if m.get("role") != "system"
        ]
        if not turns:
            raise LabelingError("Converse requires at least one non-system message")

        inference_config: dict[str, Any] = {"maxTokens": self._max_tokens}
        if temperature is not None:
            inference_config["temperature"] = temperature

        text = await asyncio.to_thread(
            self._converse, model, turns, system_prompts, inference_config
        )
        return _Response(choices=[_ResponseChoice(message=_Message(content=text))])

    def _converse(
        self,
        model: str,
        turns: list[dict[str, Any]],
        system_prompts: list[str],
        inference_config: dict[str, Any],
    ) -> str:
        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": turns,
            "inferenceConfig": inference_config,
        }
        if system_prompts:
            kwargs["system"] = [{"text": prompt} for prompt in system_prompts]

        try:
            response = self._client.converse(**kwargs)
        except Exception as exc:
            raise LabelingError(f"Bedrock Converse request failed: {exc}") from exc

        # A truncated response is almost always invalid JSON. Saying so beats letting
        # the caller report "not valid JSON" and hiding the real cause.
        if response.get("stopReason") == "max_tokens":
            raise LabelingError(
                f"Bedrock Converse hit the {inference_config['maxTokens']}-token output "
                "limit; raise BEDROCK_LABELING_MAX_TOKENS"
            )

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(block["text"] for block in blocks if "text" in block)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


class BedrockClusterLabeler:
    def __init__(
        self,
        settings: Settings,
        *,
        chat_completions_api: ChatCompletionsAPI | None = None,
    ) -> None:
        if chat_completions_api is None and not settings.aws_region:
            raise LabelingConfigurationError("AWS_REGION is required")

        if chat_completions_api is None and not settings.bedrock_labeling_model:
            raise LabelingConfigurationError("BEDROCK_LABELING_MODEL is required")

        self._model = settings.bedrock_labeling_model
        self._temperature = settings.bedrock_labeling_temperature
        self._sample_size = settings.llm_cluster_sample_size
        self._sample_max_chars = settings.llm_sample_max_chars
        self._owned_api: BedrockConverseAPI | None = None

        if chat_completions_api is None:
            self._owned_api = BedrockConverseAPI(
                region=settings.aws_region,
                max_tokens=settings.bedrock_labeling_max_tokens,
            )
            self._chat_completions_api: ChatCompletionsAPI = self._owned_api
        else:
            self._chat_completions_api = chat_completions_api

    async def aclose(self) -> None:
        if self._owned_api is None:
            return

        close = getattr(self._owned_api, "aclose", None) or getattr(
            self._owned_api, "close", None
        )
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def label_cluster(self, cluster_id: int, points: list[StoryPoint]) -> ClusterTheme:
        samples = [point.text[: self._sample_max_chars] for point in points[: self._sample_size]]
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You label clusters of story chunks. Return compact JSON with "
                        'keys "theme" and "description". The theme must be 2-6 words.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Cluster {cluster_id} contains these text chunks:\n\n"
                        + "\n\n---\n\n".join(samples)
                    ),
                },
            ],
        }
        # Some model families reject the temperature parameter outright, even set to
        # their own default - so a None setting must omit it from the call entirely
        # rather than pass a value.
        if self._temperature is not None:
            create_kwargs["temperature"] = self._temperature

        response = cast(
            _ChatResponse,
            await self._chat_completions_api.create(**create_kwargs),
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LabelingError(f"empty label response for cluster {cluster_id}")

        return _parse_cluster_theme(content)


def _strip_code_fence(content: str) -> str:
    """Remove a Markdown fence around a JSON body.

    Nova Pro returns bare JSON, but Nova Lite and Micro wrap it in ```json fences
    and a model swap should not break labelling. Cheap to tolerate, expensive to
    debug if it ever happens in production.
    """
    text = content.strip()
    if not text.startswith("```"):
        return text

    without_open = text[3:]
    if without_open.lower().startswith("json"):
        without_open = without_open[4:]
    return without_open.removesuffix("```").strip()


def _parse_cluster_theme(content: str) -> ClusterTheme:
    try:
        payload = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        raise LabelingError("label response was not valid JSON") from exc

    theme = payload.get("theme")
    description = payload.get("description")
    if not isinstance(theme, str) or not theme.strip():
        raise LabelingError("label response missing theme")

    normalized_description = description.strip() if isinstance(description, str) else None
    return ClusterTheme(
        theme=theme.strip(),
        description=normalized_description or None,
    )

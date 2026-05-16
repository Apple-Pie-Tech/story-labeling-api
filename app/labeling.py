from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from openai import AsyncAzureOpenAI

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
        temperature: float,
        response_format: dict[str, str],
    ) -> object: ...


class _ChoiceMessage(Protocol):
    content: str | None


class _Choice(Protocol):
    message: _ChoiceMessage


class _ChatResponse(Protocol):
    choices: list[_Choice]


class AzureClusterLabeler:
    def __init__(
        self,
        settings: Settings,
        *,
        chat_completions_api: ChatCompletionsAPI | None = None,
    ) -> None:
        if chat_completions_api is None and not settings.azure_openai_endpoint:
            raise LabelingConfigurationError("AZURE_OPENAI_ENDPOINT is required")

        if chat_completions_api is None and not settings.azure_openai_api_key:
            raise LabelingConfigurationError("AZURE_OPENAI_API_KEY is required")

        self._deployment = settings.azure_openai_labeling_deployment
        self._sample_size = settings.llm_cluster_sample_size
        self._sample_max_chars = settings.llm_sample_max_chars
        self._client: AsyncAzureOpenAI | None = None

        if chat_completions_api is None:
            self._client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
            )
            self._chat_completions_api = self._client.chat.completions
        else:
            self._chat_completions_api = chat_completions_api

    async def aclose(self) -> None:
        if self._client is None:
            return

        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def label_cluster(self, cluster_id: int, points: list[StoryPoint]) -> ClusterTheme:
        samples = [point.text[: self._sample_max_chars] for point in points[: self._sample_size]]
        response = cast(
            _ChatResponse,
            await self._chat_completions_api.create(
                model=self._deployment,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
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
            ),
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LabelingError(f"empty label response for cluster {cluster_id}")

        return _parse_cluster_theme(content)


def _parse_cluster_theme(content: str) -> ClusterTheme:
    try:
        payload = json.loads(content)
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

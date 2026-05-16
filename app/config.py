from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "apple_pie_story_chunks"
    qdrant_scroll_batch_size: int = 256
    qdrant_update_batch_size: int = 64

    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = None

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_labeling_deployment: str = "gpt-4o-mini"

    llm_cluster_sample_size: int = 8
    llm_sample_max_chars: int = 700


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # S3 Vectors bucket and index, both created by Terraform. The index pins the
    # dimension and the non-filterable metadata keys at creation time.
    s3_vector_bucket: str | None = None
    s3_vector_index: str = "apple-pie-story-chunks"
    # Capped by the service at its own limits: ListVectors returns at most 1000
    # per page, PutVectors accepts at most 500 per call.
    vector_list_batch_size: int = 500
    vector_put_batch_size: int = 500

    hdbscan_min_cluster_size: int = 10
    # None tells hdbscan to fall back to its own default (min_cluster_size).
    hdbscan_min_samples: int | None = None

    aws_region: str = "us-east-1"
    # Nova Pro, not an OpenAI or Claude model: the GPT-6/GPT-5.6 families and every
    # Claude 5.x model are listed in this account's Bedrock catalogue but return
    # "not available for this account" when invoked, and Claude additionally needs
    # the Anthropic use-case form. Nova Pro was verified to return bare, parseable
    # JSON for both of this stack's prompts.
    bedrock_labeling_model: str = "amazon.nova-pro-v1:0"
    # None omits the `temperature` parameter from the call entirely, which some
    # model families require - they reject any explicit value, including the default.
    bedrock_labeling_temperature: float | None = 0.1
    # A truncated response is invalid JSON, so this is a correctness setting, not
    # just a cost one; BedrockConverseAPI raises explicitly when it is hit.
    bedrock_labeling_max_tokens: int = 512

    @field_validator("hdbscan_min_samples", "bedrock_labeling_temperature", mode="before")
    @classmethod
    def _blank_env_var_means_default(cls, value: object) -> object:
        """Treat a blank numeric env var (e.g. `KEY=` in .env) as unset.

        A blank env var is a realistic deployment accident (a template
        placeholder left empty), not a real number, and pydantic won't
        coerce "" to an int/float on its own. Fall back to the field's
        documented None default rather than hard-failing Settings validation.
        """
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    llm_cluster_sample_size: int = 8
    llm_sample_max_chars: int = 700


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

# Apple Pie Story Labeling API

Minimal API that clusters Qdrant story chunk embeddings with HDBSCAN, labels each cluster with Azure OpenAI, and writes the label metadata back to Qdrant.

## Endpoints

- `GET /health`
- `POST /cluster-labels`

## Configuration

```text
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=apple_pie_story_chunks
QDRANT_SCROLL_BATCH_SIZE=256
QDRANT_UPDATE_BATCH_SIZE=64
HDBSCAN_MIN_CLUSTER_SIZE=10
HDBSCAN_MIN_SAMPLES=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_LABELING_DEPLOYMENT=gpt-4o-mini
LLM_CLUSTER_SAMPLE_SIZE=8
LLM_SAMPLE_MAX_CHARS=700
```

## Local Development

```bash
uv run --python 3.12 --with-editable . --with pytest --with pytest-asyncio --with httpx pytest
uv run uvicorn app.main:app --reload --port 8001
```

Run a labeling job:

```bash
curl -X POST http://localhost:8001/cluster-labels
```

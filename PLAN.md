# Minimal Story Clustering API Plan

## Summary

Build a new FastAPI service in `story-labeling-api/`, next to `data-ingestion/`, that reads all existing Qdrant points from `apple_pie_story_chunks`, clusters their original embedding vectors directly with HDBSCAN, asks an LLM for a short theme label per cluster, and writes the cluster metadata back into each point payload.

## Key Changes

- Create a standalone Python service using `FastAPI`, `uvicorn`, `pydantic-settings`, `qdrant-client`, `numpy`, `hdbscan`, and `openai`.
- Mirror the ingestion service shape with `app/config.py`, `app/vector_store.py`, `app/clustering.py`, `app/labeling.py`, `app/service.py`, and `app/main.py`.
- Add `GET /health` and `POST /cluster-labels`.
- Use the existing Qdrant payload shape from `data-ingestion/app/vector_store.py`: each point already has `text`, `input_id`, `user_id`, `timestamp`, `chunk_index`, `source`, and `embedding_model`.

## Implementation Details

- Read all points from the configured Qdrant collection with vectors and payloads included.
- Skip records missing an unnamed vector or usable `payload["text"]`.
- Cluster the original embedding matrix directly, without UMAP or PCA preprocessing.
- Use `hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES)`.
- Treat HDBSCAN label `-1` as noise/outliers and write noise metadata without an LLM call.
- For each non-noise cluster, send up to `LLM_CLUSTER_SAMPLE_SIZE` text chunks to Azure OpenAI and ask for compact JSON containing `theme` and `description`.
- Preserve existing Qdrant payload fields and add one `clustering` payload object:

```json
{
  "algorithm": "hdbscan",
  "scope": "full_collection_original_embedding_space",
  "cluster_id": 0,
  "theme": "example theme",
  "description": "optional short description",
  "is_noise": false
}
```

- For every non-noise cluster, define the centroid as the median value in each original embedding dimension across all points in that cluster.
- Save each centroid as a separate deterministic Qdrant point with id `centroid:hdbscan:<cluster_id>`, vector equal to the median centroid, and payload containing `is_centroid: true`, `cluster_id`, `theme`, and `description`.
- Ignore existing centroid points while loading points for clustering, then replace all existing centroid points on each run so they reflect the latest full collection.
- If LLM labeling fails for a cluster, still write the numeric cluster id with fallback theme `"Cluster <id>"`.

## API Response

`POST /cluster-labels` returns:

```json
{
  "status": "completed",
  "points_read": 123,
  "points_clustered": 120,
  "clusters_found": 5,
  "noise_points": 8,
  "points_updated": 125
}
```

`points_updated` counts both the clustering payload updates written onto existing points and any centroid points re-upserted during the same run.

## Test Plan

- Unit test Qdrant point extraction from fake scroll results, including skipped records.
- Unit test HDBSCAN wrapper error handling for too few points.
- Unit test payload update shape for normal clusters and noise points.
- Unit test LLM label fallback when the provider raises.
- Add a lightweight `/health` API test.

## Assumptions

- One run clusters all valid points in the configured Qdrant collection.
- Clustering happens in the original embedding space, not UMAP-reduced space.
- Labels are saved as both machine-readable `cluster_id` and human-readable `theme`.
- This is a hackathon MVP, so `POST /cluster-labels` runs synchronously.

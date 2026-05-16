from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from app.clustering import ClusteringError
from app.config import Settings, get_settings
from app.labeling import LabelingError
from app.schemas import ClusterLabelResult
from app.service import ClusterLabelingService
from app.vector_store import VectorStoreError

app = FastAPI(title="Apple Pie Story Labeling API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def get_cluster_labeling_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[ClusterLabelingService]:
    service = ClusterLabelingService(settings)
    try:
        yield service
    finally:
        await service.aclose()


@app.post("/cluster-labels", response_model=ClusterLabelResult)
async def cluster_labels(
    service: Annotated[ClusterLabelingService, Depends(get_cluster_labeling_service)],
) -> ClusterLabelResult:
    try:
        return await service.run()
    except ClusteringError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LabelingError as exc:
        raise HTTPException(status_code=502, detail="labeling_unavailable") from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail="vector_store_unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="internal_server_error") from exc

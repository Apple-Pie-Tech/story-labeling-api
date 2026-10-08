import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

from app.clustering import ClusteringError
from app.config import Settings, get_settings
from app.labeling import LabelingError
from app.schemas import ClusterLabelResult
from app.service import ClusterLabelingService
from app.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

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
    x_ingest_input_id: Annotated[str | None, Header()] = None,
    x_ingest_source: Annotated[str | None, Header()] = None,
) -> ClusterLabelResult:
    # This endpoint always reclusters the full collection, so these headers
    # don't scope the work; they're accepted purely to trace which ingest
    # run triggered a given labeling pass.
    if x_ingest_input_id or x_ingest_source:
        logger.info(
            "cluster-labels triggered by ingest input_id=%s source=%s",
            x_ingest_input_id,
            x_ingest_source,
        )

    try:
        return await service.run()
    except ClusteringError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LabelingError as exc:
        raise HTTPException(status_code=502, detail="labeling_unavailable") from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail="vector_store_unavailable") from exc
    except Exception as exc:
        # Log the traceback before collapsing to an opaque 500. Without this an
        # unexpected failure is invisible: the client sees "internal_server_error"
        # and the service log shows nothing but the access line.
        logger.exception("cluster-labels failed with an unhandled error")
        raise HTTPException(status_code=500, detail="internal_server_error") from exc

from pydantic import BaseModel


class ClusterLabelResult(BaseModel):
    status: str
    points_read: int
    points_clustered: int
    clusters_found: int
    noise_points: int
    points_updated: int

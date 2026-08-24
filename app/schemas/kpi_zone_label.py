"""Request/response shapes for the zone-label drawing endpoints
(app.api.v1.endpoints.kpi_labels / app.services.kpi_label_service)."""
from pydantic import BaseModel, field_validator


class CameraFrameResponse(BaseModel):
    camera_id: str
    frame_base64: str
    frame_width: int
    frame_height: int


class KpiZoneLabelIn(BaseModel):
    kpi_name: str
    points: list[list[float]]

    @field_validator("points")
    @classmethod
    def _valid_polygon(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError("points must contain at least 3 [x, y] pairs to form a polygon.")
        for p in v:
            if len(p) != 2:
                raise ValueError(f"each point must be a [x, y] pair, got {p!r}.")
        return v


class SaveCameraLabelsRequest(BaseModel):
    labels: list[KpiZoneLabelIn]


class SavedKpiZoneLabel(BaseModel):
    kpi_name: str
    points: list[list[float]]
    updated_at: str


class SaveCameraLabelsResponse(BaseModel):
    camera_id: str
    labels: list[SavedKpiZoneLabel]

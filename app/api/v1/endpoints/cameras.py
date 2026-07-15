from fastapi import APIRouter, HTTPException

from app import db
from app.config_loader import get_kpi_registry
from app.email_crypto import EmailCryptoNotConfigured, encrypt_secret
from app.kpis import list_registered_names
from app.schemas import CameraCreate, CameraInfo, CameraKPIDetail, CameraListItem, CameraListResponse, CameraUpdate
from app.stream_recorder import stream_recorder_manager

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

_KPI_LABELS: dict[int, str] = {
    1: "Unauthorized Access",          2: "ANPR Detection",
    3: "Intrusion Detection",          4: "Vehicle Detection",
    5: "Guard Tour / Hand Detection",  6: "Industrial PPE",
    7: "Fire / Smoke Detection",       8: "Abandoned Object",
    9: "Missing Object",               10: "Intrusion",
    11: "People Count",                12: "Engagement Time / Unattended Guest",
    13: "Staff Absence",               14: "Uniform Detection",
    15: "Carton / Box Detection",      16: "Occupancy",
    17: "Usage Time",                  18: "Fallen Object",
    19: "Guard Presence",              20: "Falling Pose",
    21: "Camera Tampering",            22: "Loading / Unloading Detection",
    23: "Cleaning as per Schedule",    24: "People Density Measurement",
    25: "Opening / Closing Time",      26: "Pack Count",
    27: "Occupancy Count & Dwell Time",28: "Detect Object",
    29: "Smoking",                     30: "Floating Object / Man Overboard",
}


def _camera_to_info(cam: db.Camera) -> CameraInfo:
    registry = get_kpi_registry()
    implemented_names = set(list_registered_names())
    kpis_detail = [
        CameraKPIDetail(
            kpi_id=kid,
            kpi_label=_KPI_LABELS.get(kid, f"KPI {kid}"),
            implemented=registry.get(str(kid)) in implemented_names,
        )
        for kid in cam.kpi_ids
    ]
    stream_status = stream_recorder_manager.status_for(cam.camera_id)
    return CameraInfo(
        camera_id=cam.camera_id, name=cam.name, zone=cam.zone,
        priority=cam.priority, kpi_ids=cam.kpi_ids, kpis=kpis_detail,
        camera_ip=cam.camera_ip, rtsp_port=cam.rtsp_port,
        stream_username=cam.stream_username, stream_path=cam.stream_path,
        stream_password_set=bool(cam.stream_password_encrypted),
        recording_enabled=cam.recording_enabled,
        stream_status=stream_status["status"], stream_error=stream_status["error"],
    )


@router.get("", response_model=CameraListResponse)
async def list_cameras():
    registry = get_kpi_registry()
    implemented_names = set(list_registered_names())
    items = [
        CameraListItem(
            camera_id=cam.camera_id, name=cam.name, zone=cam.zone,
            priority=cam.priority, total_kpis=len(cam.kpi_ids),
            implemented_kpis=sum(
                1 for kid in cam.kpi_ids
                if registry.get(str(kid)) in implemented_names
            ),
            recording_enabled=cam.recording_enabled,
            stream_status=stream_recorder_manager.status_for(cam.camera_id)["status"],
        )
        for cam in db.list_cameras()
    ]
    return CameraListResponse(count=len(items), cameras=items)


@router.get("/{camera_id}", response_model=CameraInfo)
async def get_camera(camera_id: str):
    cam = db.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return _camera_to_info(cam)


@router.post("", response_model=CameraInfo, status_code=201)
async def create_camera(body: CameraCreate):
    encrypted_pw = None
    if body.stream_password:
        try:
            encrypted_pw = encrypt_secret(body.stream_password)
        except EmailCryptoNotConfigured as e:
            raise HTTPException(status_code=500, detail=str(e))
    try:
        cam = db.create_camera(
            camera_id=body.camera_id, name=body.name,
            zone=body.zone, priority=body.priority, kpi_ids=body.kpi_ids,
            camera_ip=body.camera_ip, rtsp_port=body.rtsp_port,
            stream_username=body.stream_username,
            stream_password_encrypted=encrypted_pw,
            stream_path=body.stream_path, recording_enabled=body.recording_enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    stream_recorder_manager.sync_camera(cam.camera_id)
    return _camera_to_info(cam)


@router.put("/{camera_id}", response_model=CameraInfo)
async def update_camera(camera_id: str, body: CameraUpdate):
    encrypted_pw = None
    if body.stream_password:
        try:
            encrypted_pw = encrypt_secret(body.stream_password)
        except EmailCryptoNotConfigured as e:
            raise HTTPException(status_code=500, detail=str(e))
    cam = db.update_camera(
        camera_id, name=body.name, zone=body.zone,
        priority=body.priority, kpi_ids=body.kpi_ids,
        camera_ip=body.camera_ip, rtsp_port=body.rtsp_port,
        stream_username=body.stream_username,
        stream_password_encrypted=encrypted_pw,
        stream_path=body.stream_path, recording_enabled=body.recording_enabled,
    )
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    stream_recorder_manager.sync_camera(camera_id)
    return _camera_to_info(cam)


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: str):
    if not db.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    stream_recorder_manager.sync_camera(camera_id)

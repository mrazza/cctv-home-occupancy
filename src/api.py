import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.config import CONFIG
from src.database import DatabaseManager
from src.pipeline import FrameRegistry

app = FastAPI(
    title="House Presence Monitoring API",
    description="API to query real-time occupancy and enter/exit events from local CCTV feed.",
    version="0.1.0"
)

# Initialize Database Manager
db_manager = DatabaseManager(CONFIG.db_path)

# Ensure snapshot directory exists
os.makedirs(CONFIG.snapshot_dir, exist_ok=True)
# Mount snapshots as static files so they can be retrieved over HTTP/HTTPS by Mattermost
app.mount("/snapshots", StaticFiles(directory=CONFIG.snapshot_dir), name="snapshots")


class StatusResponse(BaseModel):
    is_someone_home: bool = Field(..., description="Whether anyone is currently inside the house")
    current_occupancy: int = Field(..., description="Estimated number of occupants inside")
    last_updated: str = Field(..., description="ISO Timestamp of the last state change")
    last_processed_frame: Optional[str] = Field(default=None, description="ISO Timestamp of the last processed frame from the camera stream")
    system_state: str = Field(..., description="Current system state: 'IDLE' (low-CPU OpenCV motion detection), 'ACTIVE' (full YOLOv8 object detection), or 'OFFLINE' if pipeline is not running")
    current_session_id: Optional[str] = Field(default=None, description="UUID of the current active YOLO session")


class EventLogItem(BaseModel):
    id: int
    event_type: str
    tracker_id: Optional[int]
    confidence: Optional[float]
    timestamp: str
    snapshot_path: Optional[str]
    session_id: Optional[str] = None


class ResetRequest(BaseModel):
    is_someone_home: bool = Field(..., description="Corrected presence state")
    current_occupancy: int = Field(0, ge=0, description="Corrected occupancy count")


@app.get("/status", response_model=StatusResponse, tags=["State"])
def get_status():
    """Retrieves the current presence state of the household."""
    try:
        state = db_manager.get_current_state()
        
        from src.pipeline import OrchestratorRegistry
        orchestrator = OrchestratorRegistry.get_instance()
        system_state = orchestrator.state if orchestrator is not None else "OFFLINE"
        current_session_id = orchestrator.object_tracker.session_id if orchestrator is not None else None
        
        return StatusResponse(
            is_someone_home=state["is_someone_home"],
            current_occupancy=state["current_occupancy"],
            last_updated=state["last_updated"] or "",
            last_processed_frame=FrameRegistry.get_last_timestamp_iso(),
            system_state=system_state,
            current_session_id=current_session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/events", response_model=List[EventLogItem], tags=["State"])
def get_events(limit: int = Query(default=10, ge=1, le=100)):
    """Retrieves the list of recent presence transitions."""
    try:
        events = db_manager.get_recent_events(limit=limit)
        return [EventLogItem(**e) for e in events]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/reset", tags=["State"])
def manual_reset(payload: ResetRequest):
    """
    Manually overrides/reconciles the household occupancy state.
    Use this if state tracking gets out of sync (e.g. multiple people crossing together).
    """
    try:
        event_id = db_manager.force_reset_state(
            is_someone_home=payload.is_someone_home,
            current_occupancy=payload.current_occupancy
        )
        return {
            "status": "success",
            "message": "Presence state manually reset",
            "event_id": event_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/snapshot/{filename}", tags=["Snapshots"])
def get_snapshot(filename: str):
    """Serves a captured face/body crop snapshot directly."""
    filepath = os.path.join(CONFIG.snapshot_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(filepath)


def calculate_arrow_endpoint(A: tuple[int, int], B: tuple[int, int], length: float = 50.0) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Given a line segment from A to B, calculates the center point of AB
    and an endpoint for an arrow pointing to the left-hand side of vector AB.
    """
    Ax, Ay = A
    Bx, By = B
    
    # Midpoint of AB
    Cx = int((Ax + Bx) / 2)
    Cy = int((Ay + By) / 2)
    
    # Vector AB
    vx = Bx - Ax
    vy = By - Ay
    
    # Perpendicular vector pointing to the left side: (-vy, vx)
    nx = -vy
    ny = vx
    
    # Normalize perpendicular vector
    mag = np.sqrt(nx**2 + ny**2)
    if mag < 1e-9:
        return (Cx, Cy), (Cx, Cy)
        
    ux = nx / mag
    uy = ny / mag
    
    # Arrow destination
    Dx = int(Cx + length * ux)
    Dy = int(Cy + length * uy)
    
    return (Cx, Cy), (Dx, Dy)


@app.get("/frame", tags=["Stream"])
def get_current_frame(
    draw_tripwire: bool = Query(default=False, description="Draw the tripwire line and inside direction vector on the frame"),
    draw_roi: bool = Query(default=False, description="Draw the motion detection Region of Interest (ROI) polygon/bounding box on the frame"),
    width: Optional[int] = Query(default=None, ge=1, description="Optional target width for resizing"),
    height: Optional[int] = Query(default=None, ge=1, description="Optional target height for resizing")
):
    """Retrieves the current real-time frame from the video stream."""
    # 1. Fetch from shared frame registry
    frame = FrameRegistry.get_frame()
    
    # 2. Fallback if pipeline is not running
    if frame is None:
        rtsp_url = CONFIG.rtsp_url
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail="Camera stream is offline or unavailable.")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise HTTPException(status_code=503, detail="Failed to retrieve frame from camera stream.")

    # 3. Optional: Draw Tripwire Overlay
    if draw_tripwire:
        h, w, _ = frame.shape
        tx1, ty1 = int(CONFIG.tripwire_line[0][0] * w), int(CONFIG.tripwire_line[0][1] * h)
        tx2, ty2 = int(CONFIG.tripwire_line[1][0] * w), int(CONFIG.tripwire_line[1][1] * h)
        pt_A = (tx1, ty1)
        pt_B = (tx2, ty2)
        
        # Draw tripwire line
        cv2.line(frame, pt_A, pt_B, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(frame, pt_A, 6, (0, 0, 255), -1)
        cv2.circle(frame, pt_B, 6, (255, 0, 0), -1)
        
        # Draw inside vector arrow
        mid, arrow_dest = calculate_arrow_endpoint(pt_A, pt_B)
        cv2.arrowedLine(frame, mid, arrow_dest, (0, 255, 0), 3, tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.putText(frame, "INSIDE / ENTER", (arrow_dest[0] + 10, arrow_dest[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # 3b. Optional: Draw ROI Overlay
    if draw_roi and CONFIG.motion_roi is not None and len(CONFIG.motion_roi) >= 2:
        h, w, _ = frame.shape
        if len(CONFIG.motion_roi) == 2:
            # Rectangle / bounding box ROI
            rx1 = int(CONFIG.motion_roi[0][0] * w)
            ry1 = int(CONFIG.motion_roi[0][1] * h)
            rx2 = int(CONFIG.motion_roi[1][0] * w)
            ry2 = int(CONFIG.motion_roi[1][1] * h)
            x_min, x_max = sorted([rx1, rx2])
            y_min, y_max = sorted([ry1, ry2])
            
            # Semi-transparent overlay inside rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            # Outline
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2, cv2.LINE_AA)
        else:
            # Polygon ROI
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in CONFIG.motion_roi], dtype=np.int32)
            
            # Semi-transparent overlay inside polygon
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            # Outline
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

    # 4. Optional: Resize frame
    if width or height:
        h, w, _ = frame.shape
        target_w = width or int(w * (height / h))
        target_h = height or int(h * (width / w))
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 5. Encode to JPEG
    ret, jpeg = cv2.imencode(".jpg", frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame as JPEG.")
        
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@app.post("/webhook/trigger_motion", tags=["Trigger"])
def trigger_motion_event():
    """Endpoint for Scrypted/external webhooks to trigger on-demand stream processing upon motion detection."""
    if CONFIG.trigger_mode != "event":
        raise HTTPException(status_code=400, detail="System is not in on-demand 'event' trigger mode.")
    
    from src.pipeline import OrchestratorRegistry
    orchestrator = OrchestratorRegistry.get_instance()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="CCTV monitoring orchestrator is not running or initialized.")
    
    orchestrator.trigger_event_window(duration=CONFIG.event_stream_duration)
    return {
        "status": "success",
        "message": "On-demand stream processing triggered",
        "active_until": orchestrator.active_until
    }

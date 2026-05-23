import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.config import CONFIG
from src.database import DatabaseManager

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


class EventLogItem(BaseModel):
    id: int
    event_type: str
    tracker_id: Optional[int]
    confidence: Optional[float]
    timestamp: str
    snapshot_path: Optional[str]


class ResetRequest(BaseModel):
    is_someone_home: bool = Field(..., description="Corrected presence state")
    current_occupancy: int = Field(0, ge=0, description="Corrected occupancy count")


@app.get("/status", response_model=StatusResponse, tags=["State"])
def get_status():
    """Retrieves the current presence state of the household."""
    try:
        state = db_manager.get_current_state()
        return StatusResponse(
            is_someone_home=state["is_someone_home"],
            current_occupancy=state["current_occupancy"],
            last_updated=state["last_updated"] or ""
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

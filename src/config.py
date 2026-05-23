import os
from pydantic import BaseModel, Field

class CameraConfig(BaseModel):
    rtsp_url: str = Field(default="rtsp://localhost:8554/nest-cam", description="RTSP URL for the camera stream")
    fps_limit: int = Field(default=10, description="Target frames per second to process")
    
    # Motion Detection Parameters (Fast Stage)
    motion_threshold: float = Field(default=0.005, description="Fraction of frame pixels changed to trigger motion (0.0 to 1.0)")
    motion_min_contour_area: int = Field(default=500, description="Minimum contour area for motion detection")
    background_alpha: float = Field(default=0.05, description="Accumulator update speed for background subtraction")
    motion_cooldown_frames: int = Field(default=150, description="Number of idle frames before returning to fast motion detection stage")

    # Line Crossing Parameters (Slow Stage)
    # Tripwire Line: list of tuples [(x1, y1), (x2, y2)] representing the door threshold.
    # Coordinates are normalized (0.0 to 1.0) so they don't depend on resolution.
    tripwire_line: list[tuple[float, float]] = Field(
        default=[(0.2, 0.5), (0.8, 0.5)],
        description="Coordinates of the tripwire line segment [(x1, y1), (x2, y2)] normalized between 0.0 and 1.0"
    )
    
    # Database Settings
    db_path: str = Field(default="db/presence.db", description="Path to SQLite database file")
    
    # Snapshot Capture settings
    snapshot_dir: str = Field(default="snapshots", description="Directory to save person/face croppings")

# Load settings from environment variables if present
CONFIG = CameraConfig(
    rtsp_url=os.getenv("CCTV_RTSP_URL", "rtsp://localhost:8554/nest-cam"),
    fps_limit=int(os.getenv("CCTV_FPS_LIMIT", "10")),
    motion_threshold=float(os.getenv("CCTV_MOTION_THRESHOLD", "0.005")),
    motion_min_contour_area=int(os.getenv("CCTV_MIN_CONTOUR_AREA", "500")),
    background_alpha=float(os.getenv("CCTV_BACKGROUND_ALPHA", "0.05")),
    motion_cooldown_frames=int(os.getenv("CCTV_MOTION_COOLDOWN", "150")),
    db_path=os.getenv("CCTV_DB_PATH", "db/presence.db"),
    snapshot_dir=os.getenv("CCTV_SNAPSHOT_DIR", "snapshots")
)

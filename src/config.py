import os
import json
from typing import Optional
from pydantic import BaseModel, Field

class CameraConfig(BaseModel):
    rtsp_url: str = Field(default="rtsp://localhost:8554/nest-cam", description="RTSP URL for the camera stream")
    fps_limit: int = Field(default=10, description="Target frames per second to process")
    
    # API Server Settings
    host: str = Field(default="0.0.0.0", description="IP address to bind the API server to")
    port: int = Field(default=8000, description="Port to bind the API server to")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_file: Optional[str] = Field(default="logs/cctv.log", description="Path to log file (set to null/None to disable file logging)")
    
    # Trigger Mode
    trigger_mode: str = Field(default="continuous", description="Trigger mode: 'continuous' or 'event'")
    event_stream_duration: int = Field(default=45, description="Duration in seconds to keep stream active after a trigger event")
    
    # Motion Detection Parameters (Fast Stage)
    motion_threshold: float = Field(default=0.005, description="Fraction of frame pixels changed to trigger motion (0.0 to 1.0)")
    motion_min_contour_area: int = Field(default=500, description="Minimum contour area for motion detection")
    background_alpha: float = Field(default=0.05, description="Accumulator update speed for background subtraction")
    motion_cooldown_frames: int = Field(default=150, description="Number of idle frames before returning to fast motion detection stage")
    motion_roi: Optional[list[tuple[float, float]]] = Field(
        default=None,
        description="Region of interest for motion detection as [(x1, y1), (x2, y2)] normalized between 0.0 and 1.0. If None, the entire frame is used."
    )

    # Line Crossing Parameters (Slow Stage)
    # Tripwire Line: list of tuples [(x1, y1), (x2, y2)] representing the door threshold.
    # Coordinates are normalized (0.0 to 1.0) so they don't depend on resolution.
    tripwire_line: list[tuple[float, float]] = Field(
        default=[(0.2, 0.5), (0.8, 0.5)],
        description="Coordinates of the tripwire line segment [(x1, y1), (x2, y2)] normalized between 0.0 and 1.0"
    )
    tripwire_dead_zone_width: float = Field(
        default=0.05,
        description="Width of the hysteresis dead zone around the tripwire, as a fraction of frame height (0.0 to 1.0). "
                    "The centroid must move beyond half this width on the far side of the line to register a crossing. "
                    "Set to 0.0 to disable the dead zone."
    )
    tracker_confidence: float = Field(
        default=0.1,
        description="Minimum detection confidence threshold for YOLO person detections (0.0 to 1.0). "
                    "Lower values catch more detections but may increase false positives."
    )
    track_buffer: int = Field(
        default=30,
        description="Number of frames to keep lost tracks alive before reassigning a new tracker ID. "
                    "At 10 FPS, the default of 30 means tracks survive 3 seconds of occlusion."
    )
    
    # Database Settings
    db_path: str = Field(default="db/presence.db", description="Path to SQLite database file")
    
    # Snapshot Capture settings
    snapshot_dir: str = Field(default="snapshots", description="Directory to save person/face croppings")
    
    # Webhook Settings
    webhook_urls: list[str] = Field(default_factory=list, description="List of webhook URLs to trigger on events")
    webhook_timeout: int = Field(default=5, description="Timeout in seconds for the webhook requests")

def load_config() -> CameraConfig:
    config_dict = {}
    
    # 1. Try loading from config.json if it exists
    config_json_path = os.getenv("CCTV_CONFIG_PATH", "config.json")
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    config_dict.update(loaded)
        except Exception as e:
            print(f"[-] Warning: Failed to parse config JSON: {e}")
            
    # 2. Override from Environment Variables
    env_mappings = {
        "CCTV_RTSP_URL": ("rtsp_url", str),
        "CCTV_FPS_LIMIT": ("fps_limit", int),
        "CCTV_HOST": ("host", str),
        "CCTV_PORT": ("port", int),
        "CCTV_LOG_LEVEL": ("log_level", str),
        "CCTV_LOG_FILE": ("log_file", lambda x: None if x.lower() in ("null", "none", "") else str(x)),
        "CCTV_TRIGGER_MODE": ("trigger_mode", str),
        "CCTV_EVENT_STREAM_DURATION": ("event_stream_duration", int),
        "CCTV_MOTION_THRESHOLD": ("motion_threshold", float),
        "CCTV_MIN_CONTOUR_AREA": ("motion_min_contour_area", int),
        "CCTV_BACKGROUND_ALPHA": ("background_alpha", float),
        "CCTV_MOTION_COOLDOWN": ("motion_cooldown_frames", int),
        "CCTV_DB_PATH": ("db_path", str),
        "CCTV_SNAPSHOT_DIR": ("snapshot_dir", str),
        "CCTV_WEBHOOK_TIMEOUT": ("webhook_timeout", int),
        "CCTV_DEAD_ZONE_WIDTH": ("tripwire_dead_zone_width", float),
        "CCTV_TRACKER_CONFIDENCE": ("tracker_confidence", float),
        "CCTV_TRACK_BUFFER": ("track_buffer", int),
    }
    
    # Handle CCTV_WEBHOOK_URLS env variable
    webhook_urls_env = os.getenv("CCTV_WEBHOOK_URLS")
    if webhook_urls_env is not None:
        try:
            parsed = json.loads(webhook_urls_env)
            if isinstance(parsed, list):
                config_dict["webhook_urls"] = [str(p) for p in parsed]
        except Exception:
            config_dict["webhook_urls"] = [x.strip() for x in webhook_urls_env.split(",") if x.strip()]

    # Handle CCTV_MOTION_ROI env variable
    motion_roi_env = os.getenv("CCTV_MOTION_ROI")
    if motion_roi_env is not None:
        try:
            parsed = json.loads(motion_roi_env)
            if isinstance(parsed, list):
                config_dict["motion_roi"] = [(float(p[0]), float(p[1])) for p in parsed]
        except Exception:
            try:
                floats = [float(x.strip()) for x in motion_roi_env.split(",") if x.strip()]
                if len(floats) >= 4 and len(floats) % 2 == 0:
                    config_dict["motion_roi"] = [(floats[i], floats[i+1]) for i in range(0, len(floats), 2)]
            except Exception as e:
                print(f"[-] Warning: Failed to parse CCTV_MOTION_ROI env: {e}")

    for env_var, (config_key, val_type) in env_mappings.items():
        val = os.getenv(env_var)
        if val is not None:
            try:
                config_dict[config_key] = val_type(val)
            except Exception as e:
                print(f"[-] Warning: Failed to parse environment variable {env_var}: {e}")
                
    # Handle CCTV_TRIPWIRE_LINE env variable
    tripwire_env = os.getenv("CCTV_TRIPWIRE_LINE")
    if tripwire_env is not None:
        try:
            parsed = json.loads(tripwire_env)
            if isinstance(parsed, list) and len(parsed) == 2:
                config_dict["tripwire_line"] = [(float(p[0]), float(p[1])) for p in parsed]
        except Exception:
            try:
                floats = [float(x.strip()) for x in tripwire_env.split(",") if x.strip()]
                if len(floats) == 4:
                    config_dict["tripwire_line"] = [(floats[0], floats[1]), (floats[2], floats[3])]
            except Exception as e:
                print(f"[-] Warning: Failed to parse CCTV_TRIPWIRE_LINE env: {e}")
                
    return CameraConfig(**config_dict)

CONFIG = load_config()

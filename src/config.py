import os
import json
from pydantic import BaseModel, Field

class CameraConfig(BaseModel):
    rtsp_url: str = Field(default="rtsp://localhost:8554/nest-cam", description="RTSP URL for the camera stream")
    fps_limit: int = Field(default=10, description="Target frames per second to process")
    
    # API Server Settings
    host: str = Field(default="0.0.0.0", description="IP address to bind the API server to")
    port: int = Field(default=8000, description="Port to bind the API server to")
    
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
        "CCTV_MOTION_THRESHOLD": ("motion_threshold", float),
        "CCTV_MIN_CONTOUR_AREA": ("motion_min_contour_area", int),
        "CCTV_BACKGROUND_ALPHA": ("background_alpha", float),
        "CCTV_MOTION_COOLDOWN": ("motion_cooldown_frames", int),
        "CCTV_DB_PATH": ("db_path", str),
        "CCTV_SNAPSHOT_DIR": ("snapshot_dir", str),
    }
    
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

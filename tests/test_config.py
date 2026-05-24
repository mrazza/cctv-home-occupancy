import os
import json
import pytest
from src.config import load_config, CameraConfig

def test_load_config_defaults(monkeypatch):
    # Ensure no environment variables are set that could pollute the test
    for key in list(os.environ.keys()):
        if key.startswith("CCTV_"):
            monkeypatch.delenv(key, raising=False)
    
    # Force CCTV_CONFIG_PATH to a non-existent file so config.json is not loaded
    monkeypatch.setenv("CCTV_CONFIG_PATH", "non_existent_file.json")
    
    config = load_config()
    assert isinstance(config, CameraConfig)
    assert config.rtsp_url == "rtsp://localhost:8554/nest-cam"
    assert config.fps_limit == 10
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.tripwire_line == [(0.2, 0.5), (0.8, 0.5)]

def test_load_config_env_overrides(monkeypatch):
    monkeypatch.setenv("CCTV_CONFIG_PATH", "non_existent_file.json")
    monkeypatch.setenv("CCTV_RTSP_URL", "rtsp://testing-url")
    monkeypatch.setenv("CCTV_FPS_LIMIT", "15")
    monkeypatch.setenv("CCTV_HOST", "127.0.0.1")
    monkeypatch.setenv("CCTV_PORT", "9000")
    monkeypatch.setenv("CCTV_MOTION_THRESHOLD", "0.015")
    monkeypatch.setenv("CCTV_MIN_CONTOUR_AREA", "1000")
    monkeypatch.setenv("CCTV_BACKGROUND_ALPHA", "0.1")
    monkeypatch.setenv("CCTV_MOTION_COOLDOWN", "200")
    monkeypatch.setenv("CCTV_DB_PATH", "test_db.db")
    monkeypatch.setenv("CCTV_SNAPSHOT_DIR", "test_snapshots")

    config = load_config()
    assert config.rtsp_url == "rtsp://testing-url"
    assert config.fps_limit == 15
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.motion_threshold == 0.015
    assert config.motion_min_contour_area == 1000
    assert config.background_alpha == 0.1
    assert config.motion_cooldown_frames == 200
    assert config.db_path == "test_db.db"
    assert config.snapshot_dir == "test_snapshots"

def test_load_config_tripwire_env_json(monkeypatch):
    monkeypatch.setenv("CCTV_CONFIG_PATH", "non_existent_file.json")
    
    # Valid JSON string
    monkeypatch.setenv("CCTV_TRIPWIRE_LINE", "[[0.1, 0.2], [0.3, 0.4]]")
    config = load_config()
    assert config.tripwire_line == [(0.1, 0.2), (0.3, 0.4)]

def test_load_config_tripwire_env_comma_list(monkeypatch):
    monkeypatch.setenv("CCTV_CONFIG_PATH", "non_existent_file.json")
    
    # Valid comma-separated string
    monkeypatch.setenv("CCTV_TRIPWIRE_LINE", "0.1, 0.2, 0.3, 0.4")
    config = load_config()
    assert config.tripwire_line == [(0.1, 0.2), (0.3, 0.4)]

def test_load_config_tripwire_env_invalid(monkeypatch):
    monkeypatch.setenv("CCTV_CONFIG_PATH", "non_existent_file.json")
    
    # Invalid formats should fall back to default
    monkeypatch.setenv("CCTV_TRIPWIRE_LINE", "invalid_format")
    config = load_config()
    assert config.tripwire_line == [(0.2, 0.5), (0.8, 0.5)]

    # Incorrect number of coordinates
    monkeypatch.setenv("CCTV_TRIPWIRE_LINE", "0.1, 0.2, 0.3")
    config = load_config()
    assert config.tripwire_line == [(0.2, 0.5), (0.8, 0.5)]

def test_load_config_from_file(monkeypatch, temp_dir):
    config_file_path = os.path.join(temp_dir, "config.json")
    config_data = {
        "rtsp_url": "rtsp://file-url",
        "fps_limit": 5,
        "tripwire_line": [[0.5, 0.5], [0.6, 0.6]]
    }
    with open(config_file_path, "w") as f:
        json.dump(config_data, f)
        
    monkeypatch.setenv("CCTV_CONFIG_PATH", config_file_path)
    # Ensure no environment variables override file configuration
    for key in list(os.environ.keys()):
        if key.startswith("CCTV_") and key != "CCTV_CONFIG_PATH":
            monkeypatch.delenv(key, raising=False)
            
    config = load_config()
    assert config.rtsp_url == "rtsp://file-url"
    assert config.fps_limit == 5
    assert config.tripwire_line == [(0.5, 0.5), (0.6, 0.6)]

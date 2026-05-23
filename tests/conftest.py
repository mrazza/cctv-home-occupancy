import os
import shutil
import tempfile
import pytest
from src.config import CameraConfig
from src.database import DatabaseManager

@pytest.fixture
def temp_dir():
    """Provides a temporary directory that is automatically removed after test execution."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def test_config(temp_dir):
    """Provides a CameraConfig pointed to a temporary SQLite db and snapshot folder."""
    db_file = os.path.join(temp_dir, "test_presence.db")
    snapshot_fld = os.path.join(temp_dir, "snapshots")
    return CameraConfig(
        rtsp_url="rtsp://dummy",
        db_path=db_file,
        snapshot_dir=snapshot_fld
    )

@pytest.fixture
def db_manager(test_config):
    """Provides an initialized DatabaseManager instance pointed to a temporary SQLite database."""
    manager = DatabaseManager(test_config.db_path)
    return manager

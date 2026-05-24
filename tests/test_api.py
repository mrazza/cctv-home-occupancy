import os
import pytest
from fastapi.testclient import TestClient
from src.api import app, db_manager
from src.config import CONFIG

# Inject the test DB manager into the fast API app to prevent modifying the production db during test runs.
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_db(db_manager):
    """Enforces that the database used by the API routes is the temporary test database."""
    # Temporarily override db_manager inside api module
    import src.api as api
    original_db = api.db_manager
    api.db_manager = db_manager
    yield
    api.db_manager = original_db

def test_api_get_status_initial():
    """Verifies initially the status is empty / no one home."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_someone_home"] is False
    assert data["current_occupancy"] == 0
    assert data["last_updated"] != ""  # Should be populated with initial insert timestamp


def test_api_get_events_empty():
    response = client.get("/events")
    assert response.status_code == 200
    assert response.json() == []

def test_api_manual_reset(db_manager):
    # Set to true with 3 people
    payload = {"is_someone_home": True, "current_occupancy": 3}
    response = client.post("/reset", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Query status
    status_resp = client.get("/status")
    data = status_resp.json()
    assert data["is_someone_home"] is True
    assert data["current_occupancy"] == 3
    assert data["last_updated"] != ""
    
    # Check that FORCE_RESET logged in events
    events_resp = client.get("/events")
    events = events_resp.json()
    assert len(events) == 1
    assert events[0]["event_type"] == "FORCE_RESET"

def test_api_invalid_reset_payload():
    # Negative occupancy should fail validation
    payload = {"is_someone_home": True, "current_occupancy": -1}
    response = client.post("/reset", json=payload)
    assert response.status_code == 422 # Pydantic Validation Error

def test_api_get_status_error(monkeypatch):
    """Verifies that if db_manager throws an exception, the /status endpoint returns 500."""
    import src.api as api
    def mock_get_current_state():
        raise RuntimeError("Database connection lost")
    monkeypatch.setattr(api.db_manager, "get_current_state", mock_get_current_state)
    
    response = client.get("/status")
    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]

def test_api_get_events_error(monkeypatch):
    """Verifies that if db_manager throws an exception, the /events endpoint returns 500."""
    import src.api as api
    def mock_get_recent_events(limit):
        raise RuntimeError("Database read failed")
    monkeypatch.setattr(api.db_manager, "get_recent_events", mock_get_recent_events)
    
    response = client.get("/events")
    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]

def test_api_manual_reset_error(monkeypatch):
    """Verifies that if db_manager throws an exception during force_reset_state, the /reset endpoint returns 500."""
    import src.api as api
    def mock_force_reset_state(is_someone_home, current_occupancy):
        raise RuntimeError("Database write failed")
    monkeypatch.setattr(api.db_manager, "force_reset_state", mock_force_reset_state)
    
    payload = {"is_someone_home": True, "current_occupancy": 3}
    response = client.post("/reset", json=payload)
    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]

def test_api_get_snapshot_not_found():
    """Verifies that a 404 is returned if a snapshot does not exist."""
    response = client.get("/snapshot/non_existent_snapshot.jpg")
    assert response.status_code == 404
    assert response.json()["detail"] == "Snapshot not found"

def test_api_get_snapshot_success(temp_dir, monkeypatch):
    """Verifies that a valid snapshot file is served successfully."""
    import src.api as api
    # Create a dummy image file
    snapshot_filename = "test_snapshot.jpg"
    filepath = os.path.join(temp_dir, snapshot_filename)
    with open(filepath, "wb") as f:
        f.write(b"dummy_image_data")
        
    # Temporarily override CONFIG.snapshot_dir
    monkeypatch.setattr(api.CONFIG, "snapshot_dir", temp_dir)
    
    response = client.get(f"/snapshot/{snapshot_filename}")
    assert response.status_code == 200
    assert response.content == b"dummy_image_data"

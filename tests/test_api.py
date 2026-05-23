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

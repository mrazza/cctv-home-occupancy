from src.database import DatabaseManager

def test_database_initialization(db_manager):
    """Ensures database is initialized with default schema and correct state single row."""
    state = db_manager.get_current_state()
    assert state.is_someone_home is False
    assert state.current_occupancy == 0
    assert state.last_updated != ""  # Should be populated with initial insert timestamp


def test_log_enter_event(db_manager):
    """Ensures entering increments occupancy and activates presence."""
    eid = db_manager.log_event("ENTER", tracker_id=1, confidence=0.92, session_id="test-session-123")
    assert eid > 0
    
    state = db_manager.get_current_state()
    assert state.is_someone_home is True
    assert state.current_occupancy == 1
    assert state.last_updated != ""

    events = db_manager.get_recent_events(limit=1)
    assert len(events) == 1
    assert events[0].event_type == "ENTER"
    assert events[0].tracker_id == 1
    assert events[0].confidence == 0.92
    assert events[0].session_id == "test-session-123"

def test_log_leave_event(db_manager):
    """Ensures leaving decrements occupancy correctly."""
    # First enter
    db_manager.log_event("ENTER", tracker_id=1)
    db_manager.log_event("ENTER", tracker_id=2)
    
    # Verify current occupancy
    assert db_manager.get_current_state().current_occupancy == 2
    
    # Leave 1 person
    db_manager.log_event("LEAVE", tracker_id=1)
    state = db_manager.get_current_state()
    assert state.is_someone_home is True
    assert state.current_occupancy == 1

    # Leave second person
    db_manager.log_event("LEAVE", tracker_id=2)
    state = db_manager.get_current_state()
    assert state.is_someone_home is False
    assert state.current_occupancy == 0

def test_log_leave_event_negative_safety(db_manager):
    """Ensures leaving doesn't result in negative occupancy numbers."""
    db_manager.log_event("LEAVE", tracker_id=1)
    state = db_manager.get_current_state()
    assert state.is_someone_home is False
    assert state.current_occupancy == 0

def test_force_reset_state(db_manager):
    """Ensures forced state updates function correctly and are recorded in the log."""
    db_manager.log_event("ENTER", tracker_id=1)
    
    # Force reset
    db_manager.force_reset_state(is_someone_home=True, current_occupancy=5)
    state = db_manager.get_current_state()
    assert state.is_someone_home is True
    assert state.current_occupancy == 5

    events = db_manager.get_recent_events(limit=5)
    assert events[0].event_type == "FORCE_RESET"

def test_log_event_numpy_types(db_manager):
    """Ensures that passing NumPy types (like np.int64 and np.float32) does not result in BLOB insertions."""
    import numpy as np

    # Use NumPy types
    np_tracker_id = np.int64(42)
    np_confidence = np.float32(0.85)

    eid = db_manager.log_event("ENTER", tracker_id=np_tracker_id, confidence=np_confidence)
    assert eid > 0

    # Ensure get_recent_events retrieves Python native int and float, not bytes or numpy types
    events = db_manager.get_recent_events(limit=1)
    assert len(events) == 1
    retrieved_tracker_id = events[0].tracker_id
    retrieved_confidence = events[0].confidence

    # Assert correct values and types
    assert retrieved_tracker_id == 42
    assert isinstance(retrieved_tracker_id, int)
    assert abs(retrieved_confidence - 0.85) < 1e-5
    assert isinstance(retrieved_confidence, float)


def test_database_get_current_state_empty(temp_dir):
    """Verifies that if presence_state table is queried but returns no rows, empty defaults are returned."""
    # Create a fresh db without initializing the row
    import os
    from src.database import DatabaseManager
    db_file = os.path.join(temp_dir, "empty_state.db")
    db = DatabaseManager(db_file)
    
    # Manually delete the default row that is inserted on init_db
    conn = db.get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM presence_state;")
            conn.commit()
    finally:
        conn.close()
        
    state = db.get_current_state()
    assert state.is_someone_home is False
    assert state.current_occupancy == 0
    assert state.last_updated == ""

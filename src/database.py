import os
import sqlite3
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from src.types import PresenceState, DatabaseEvent

class DatabaseManager:
    def __init__(self, db_path: str):
        """
        Initializes the DatabaseManager, creating parent directories for the SQLite file if needed,
        and triggers table initialization.
        """
        self.db_path = db_path
        # Create parent directory if needed
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """
        Establishes a connection to the SQLite database.
        Enables foreign key constraints and sets sqlite3.Row row factory for column access.
        """
        # Enable foreign keys and autocommit/concurrency tuning for SQLite
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """
        Initializes the database schema by creating presence_state and events_log tables.
        Also inserts initial default state row if not already present.
        """
        conn = self.get_connection()
        try:
            with conn:
                # Table 1: presence_state (Single row table)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS presence_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        is_someone_home BOOLEAN NOT NULL DEFAULT 0,
                        current_occupancy INTEGER NOT NULL DEFAULT 0,
                        last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Ensure the single row exists
                conn.execute("""
                    INSERT OR IGNORE INTO presence_state (id, is_someone_home, current_occupancy)
                    VALUES (1, 0, 0);
                """)

                # Table 2: events_log
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL CHECK(event_type IN ('ENTER', 'LEAVE', 'FORCE_RESET')),
                        tracker_id INTEGER,
                        confidence REAL,
                        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        snapshot_path TEXT,
                        session_id TEXT
                    );
                """)
                conn.commit()
        finally:
            conn.close()

    def get_current_state(self) -> PresenceState:
        """
        Fetches the current household presence state from the presence_state table.
        
        Returns:
            PresenceState representing current occupancy count and home/away status.
        """
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                "SELECT is_someone_home, current_occupancy, last_updated FROM presence_state WHERE id = 1"
            )
            row = cursor.fetchone()
            if row:
                return PresenceState(
                    is_someone_home=bool(row["is_someone_home"]),
                    current_occupancy=row["current_occupancy"],
                    last_updated=row["last_updated"]
                )
            return PresenceState(is_someone_home=False, current_occupancy=0, last_updated="")
        finally:
            conn.close()

    def log_event(self, event_type: str, tracker_id: Optional[int] = None, 
                  confidence: Optional[float] = None, snapshot_path: Optional[str] = None,
                  session_id: Optional[str] = None) -> int:
        """
        Logs an ENTER or LEAVE event to the database and atomically updates occupancy and presence status.
        Handles safe type casting of numpy values (int, float) before writing to the database.

        Args:
            event_type: The type of event ("ENTER", "LEAVE").
            tracker_id: Optional ID of the associated object tracker track.
            confidence: Optional confidence of the detection.
            snapshot_path: Optional file path to the saved crop snapshot.
            session_id: Optional UUID identifying the active tracker session.

        Returns:
            The row ID of the inserted event log record.
        """
        # Convert any numpy integers/floats or other convertible types to standard python types
        if tracker_id is not None:
            tracker_id = int(tracker_id)
        if confidence is not None:
            confidence = float(confidence)
            
        conn = self.get_connection()
        try:
            with conn:
                now = datetime.now().isoformat()
                
                # Determine change in occupancy
                delta = 0
                if event_type == "ENTER":
                    delta = 1
                elif event_type == "LEAVE":
                    delta = -1
                    
                # Log the event
                cursor = conn.execute("""
                    INSERT INTO events_log (event_type, tracker_id, confidence, timestamp, snapshot_path, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (event_type, tracker_id, confidence, now, snapshot_path, session_id))
                event_id = cursor.lastrowid
                
                # Update state atomically using SQLite MAX and CASE to avoid check-then-act race conditions
                conn.execute("""
                    UPDATE presence_state
                    SET current_occupancy = MAX(0, current_occupancy + ?),
                        is_someone_home = CASE WHEN MAX(0, current_occupancy + ?) > 0 THEN 1 ELSE 0 END,
                        last_updated = ?
                    WHERE id = 1
                """, (delta, delta, now))
                
                conn.commit()
                return event_id
        finally:
            conn.close()

    def force_reset_state(self, is_someone_home: bool, current_occupancy: int) -> int:
        """Forces a reset on state for reconciliation purposes and logs a FORCE_RESET event."""
        conn = self.get_connection()
        try:
            with conn:
                now = datetime.now().isoformat()
                
                # Log the reset event
                cursor = conn.execute("""
                    INSERT INTO events_log (event_type, tracker_id, confidence, timestamp, snapshot_path)
                    VALUES (?, ?, ?, ?, ?)
                """, ("FORCE_RESET", None, None, now, None))
                event_id = cursor.lastrowid
                
                # Update state
                conn.execute("""
                    UPDATE presence_state
                    SET is_someone_home = ?,
                        current_occupancy = ?,
                        last_updated = ?
                    WHERE id = 1
                """, (1 if is_someone_home else 0, max(0, current_occupancy), now))
                
                conn.commit()
                return event_id
        finally:
            conn.close()

    def get_recent_events(self, limit: int = 10) -> list[DatabaseEvent]:
        """
        Retrieves a list of recent database event log entries, sorted newest first.

        Args:
            limit: The maximum number of entries to retrieve.

        Returns:
            List of DatabaseEvent objects.
        """
        conn = self.get_connection()
        try:
            cursor = conn.execute("""
                SELECT id, event_type, tracker_id, confidence, timestamp, snapshot_path, session_id
                FROM events_log
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [
                DatabaseEvent(
                    id=row["id"],
                    event_type=row["event_type"],
                    tracker_id=row["tracker_id"],
                    confidence=row["confidence"],
                    timestamp=row["timestamp"],
                    snapshot_path=row["snapshot_path"],
                    session_id=row["session_id"]
                )
                for row in rows
            ]
        finally:
            conn.close()

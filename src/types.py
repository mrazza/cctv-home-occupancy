from dataclasses import dataclass
from typing import Optional

@dataclass
class CrossingEvent:
    event_type: str
    tracker_id: int
    confidence: float
    snapshot_path: Optional[str]


@dataclass
class PresenceState:
    is_someone_home: bool
    current_occupancy: int
    last_updated: str


@dataclass
class DatabaseEvent:
    id: int
    event_type: str
    tracker_id: Optional[int]
    confidence: Optional[float]
    timestamp: str
    snapshot_path: Optional[str]
    session_id: Optional[str] = None

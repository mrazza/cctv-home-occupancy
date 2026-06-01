import os
import cv2
import uuid
import math
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from ultralytics import YOLO

class ObjectTracker:
    # Class-level attribute to support mocking/autospecing
    session_id: Optional[str] = None

    def __init__(self, 
                 model_name: str = "yolov8n.pt", 
                 tripwire_line: Optional[List[Tuple[float, float]]] = None,
                 snapshot_dir: str = "snapshots",
                 dead_zone_width: float = 0.0):
        """
        Object Tracker utilizing YOLOv8/11 and ByteTrack to trace person trajectories and detect line crossing events.
        
        Args:
            model_name: Name of the YOLO model to use (defaults to yolov8n.pt for CPU efficiency)
            tripwire_line: Normalized tripwire line coordinates [(x1, y1), (x2, y2)]
            snapshot_dir: Directory where person crops are saved
            dead_zone_width: Width of the hysteresis dead zone around the tripwire, as a fraction of frame height.
                When set to 0.0, any signed-distance change across the line triggers an event immediately
                (note: this uses distance-based logic, not segment intersection).
        """
        # Load YOLO model
        self.model = YOLO(model_name)
        
        # Unique session ID for this instantiation
        self.session_id = str(uuid.uuid4())
        
        # Normalized tripwire line segment, e.g. [(0.2, 0.5), (0.8, 0.5)]
        self.tripwire_line = tripwire_line or [(0.2, 0.5), (0.8, 0.5)]
        self.snapshot_dir = snapshot_dir
        self.dead_zone_width = dead_zone_width
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Track history: mapping of tracker_id -> list of centroids (x, y)
        self.track_histories: Dict[int, List[Tuple[float, float]]] = {}
        # Track confirmed side state: mapping of tracker_id -> last known confirmed side (+1 or -1 or 0)
        self.track_confirmed_sides: Dict[int, int] = {}

    def _get_point_side(self, A: Tuple[float, float], B: Tuple[float, float], P: Tuple[float, float]) -> int:
        """
        Determines which side of the directed line segment AB point P is on.
        Returns:
            +1: Left/Clockwise side (designated "inside")
            -1: Right/Counterclockwise side (designated "outside")
            0: Collinear
        """
        val = (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else -1

    def _get_signed_distance(self, A: Tuple[float, float], B: Tuple[float, float],
                             P: Tuple[float, float]) -> float:
        """
        Returns the signed perpendicular distance from point P to the directed line AB.
        Positive = left/inside side (+1), Negative = right/outside side (-1).
        Result is in the same units as A, B, P (pixels when called from process_frame).
        """
        dx = B[0] - A[0]
        dy = B[1] - A[1]
        line_len = math.sqrt(dx * dx + dy * dy)
        if line_len < 1e-9:
            return 0.0
        return (dx * (P[1] - A[1]) - dy * (P[0] - A[0])) / line_len

    def save_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], tracker_id: int, event_type: str) -> Optional[str]:
        """Saves a high-resolution crop of the person bounding box for future face recognition."""
        try:
            h, w, _ = frame.shape
            x1, y1, x2, y2 = bbox
            # Clip coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{event_type.lower()}_id{tracker_id}_{timestamp}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)
            cv2.imwrite(filepath, crop)
            return filepath
        except Exception:
            return None

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object tracking on the frame and checks for tripwire crossings.
        
        Returns:
            List of detected crossing events:
            [{"event_type": "ENTER"/"LEAVE", "tracker_id": id, "confidence": conf, "snapshot_path": path}]
        """
        h, w, _ = frame.shape
        
        # Compute dead zone threshold in pixel space (based on frame height)
        dead_zone_half_px = (self.dead_zone_width * h) / 2.0
        
        # Convert normalized tripwire coordinates to pixel values
        tx1, ty1 = int(self.tripwire_line[0][0] * w), int(self.tripwire_line[0][1] * h)
        tx2, ty2 = int(self.tripwire_line[1][0] * w), int(self.tripwire_line[1][1] * h)
        A = (float(tx1), float(ty1))
        B = (float(tx2), float(ty2))

        # Run tracking. Classes=0 is 'person'.
        # persist=True ensures the tracking state is maintained.
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        
        events = []
        active_tracker_ids = set()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Extract box coordinates, tracker IDs, and confidences
            # boxes.id contains track identifiers if tracker is running
            if boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                tracker_ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for i, tracker_id in enumerate(tracker_ids):
                    active_tracker_ids.add(tracker_id)
                    x1, y1, x2, y2 = xyxy[i]
                    conf = confidences[i]
                    
                    # Centroid of bounding box to trace trajectory
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    curr_point = (cx, cy)
                    
                    # Initialize track history
                    if tracker_id not in self.track_histories:
                        self.track_histories[tracker_id] = []
                        # Calculate initial side of the tripwire
                        self.track_confirmed_sides[tracker_id] = self._get_point_side(A, B, curr_point)
                    
                    confirmed_side = self.track_confirmed_sides[tracker_id]
                    signed_dist = self._get_signed_distance(A, B, curr_point)
                    
                    event_type = None
                    # Hysteresis crossing logic:
                    # Centroid must cross to the opposite side and exceed the dead zone boundary.
                    if confirmed_side == -1 and signed_dist > dead_zone_half_px:
                        event_type = "ENTER"
                        self.track_confirmed_sides[tracker_id] = 1
                    elif confirmed_side == 1 and signed_dist < -dead_zone_half_px:
                        event_type = "LEAVE"
                        self.track_confirmed_sides[tracker_id] = -1
                    elif confirmed_side == 0:
                        # If initially collinear, commit to whichever side the centroid moves towards
                        if signed_dist > dead_zone_half_px:
                            self.track_confirmed_sides[tracker_id] = 1
                        elif signed_dist < -dead_zone_half_px:
                            self.track_confirmed_sides[tracker_id] = -1
                            
                    if event_type:
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        snapshot_path = self.save_crop(frame, bbox, tracker_id, event_type)
                        events.append({
                            "event_type": event_type,
                            "tracker_id": int(tracker_id),
                            "confidence": float(conf),
                            "snapshot_path": snapshot_path
                        })
                                
                    # Append current point to history and limit size to last 10 points
                    self.track_histories[tracker_id].append(curr_point)
                    if len(self.track_histories[tracker_id]) > 10:
                        self.track_histories[tracker_id].pop(0)

        # Cleanup old tracking histories that are no longer active to prevent memory leaks
        dead_ids = [tid for tid in self.track_histories if tid not in active_tracker_ids]
        for tid in dead_ids:
            del self.track_histories[tid]
            if tid in self.track_confirmed_sides:
                del self.track_confirmed_sides[tid]

        return events

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, 
                 model_name: str = "yolov8n.pt", 
                 tripwire_line: Optional[List[Tuple[float, float]]] = None,
                 snapshot_dir: str = "snapshots"):
        """
        Object Tracker utilizing YOLOv8/11 and ByteTrack to trace person trajectories and detect line crossing events.
        
        Args:
            model_name: Name of the YOLO model to use (defaults to yolov8n.pt for CPU efficiency)
            tripwire_line: Normalized tripwire line coordinates [(x1, y1), (x2, y2)]
            snapshot_dir: Directory where person crops are saved
        """
        # Load YOLO model
        self.model = YOLO(model_name)
        
        # Normalized tripwire line segment, e.g. [(0.2, 0.5), (0.8, 0.5)]
        self.tripwire_line = tripwire_line or [(0.2, 0.5), (0.8, 0.5)]
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Track history: mapping of tracker_id -> list of centroids (x, y)
        self.track_histories: Dict[int, List[Tuple[float, float]]] = {}
        # Track side state: mapping of tracker_id -> last known side (+1 or -1 or 0)
        self.track_sides: Dict[int, int] = {}

    def _get_ccw_orientation(self, A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> int:
        """
        Calculates orientation of triplet (A, B, C).
        Returns:
            0: Collinear
            1: Clockwise
            -1: Counterclockwise
        """
        val = (B[1] - A[1]) * (C[0] - B[0]) - (B[0] - A[0]) * (C[1] - B[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else -1

    def _check_intersection(self, A: Tuple[float, float], B: Tuple[float, float], 
                            C: Tuple[float, float], D: Tuple[float, float]) -> bool:
        """
        Checks if line segment AB and CD intersect.
        """
        o1 = self._get_ccw_orientation(A, B, C)
        o2 = self._get_ccw_orientation(A, B, D)
        o3 = self._get_ccw_orientation(C, D, A)
        o4 = self._get_ccw_orientation(C, D, B)

        # General case
        if o1 != o2 and o3 != o4:
            return True
            
        # Special cases (collinear points on segment)
        # For simplicity in movement tracking, we ignore pure collinear overlaps unless they cross.
        return False

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
                        self.track_sides[tracker_id] = self._get_point_side(A, B, curr_point)
                    
                    prev_history = self.track_histories[tracker_id]
                    
                    if len(prev_history) > 0:
                        prev_point = prev_history[-1]
                        
                        # Check intersection
                        if self._check_intersection(A, B, prev_point, curr_point):
                            # Side checking logic to verify crossing direction
                            prev_side = self.track_sides[tracker_id]
                            curr_side = self._get_point_side(A, B, curr_point)
                            
                            # If they transitioned sides
                            if prev_side != curr_side and prev_side != 0 and curr_side != 0:
                                # Define crossing event type based on direction change
                                # side = 1 is designated "inside" (Left of directed vector AB)
                                # side = -1 is designated "outside" (Right of directed vector AB)
                                if prev_side == -1 and curr_side == 1:
                                    event_type = "ENTER"
                                elif prev_side == 1 and curr_side == -1:
                                    event_type = "LEAVE"
                                else:
                                    event_type = None
                                    
                                if event_type:
                                    bbox = (int(x1), int(y1), int(x2), int(y2))
                                    snapshot_path = self.save_crop(frame, bbox, tracker_id, event_type)
                                    events.append({
                                        "event_type": event_type,
                                        "tracker_id": int(tracker_id),
                                        "confidence": float(conf),
                                        "snapshot_path": snapshot_path
                                    })
                                
                            # Update known side
                            self.track_sides[tracker_id] = curr_side
                            
                    # Append current point to history and limit size to last 10 points
                    self.track_histories[tracker_id].append(curr_point)
                    if len(self.track_histories[tracker_id]) > 10:
                        self.track_histories[tracker_id].pop(0)

        # Cleanup old tracking histories that are no longer active to prevent memory leaks
        dead_ids = [tid for tid in self.track_histories if tid not in active_tracker_ids]
        for tid in dead_ids:
            del self.track_histories[tid]
            if tid in self.track_sides:
                del self.track_sides[tid]

        return events

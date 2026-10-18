import os
import cv2
import uuid
import math
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any, Set
from src.types import CrossingEvent
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ObjectTracker:
    # Class-level attribute to support mocking/autospecing
    session_id: Optional[str] = None

    def __init__(self, 
                 model_name: str = "yolov8n.pt", 
                 tripwire_line: Optional[List[Tuple[float, float]]] = None,
                 snapshot_dir: str = "snapshots",
                 dead_zone_width: float = 0.0,
                 tripwire_strict_segment: bool = False,
                 conf: float = 0.25,
                 track_buffer: int = 30,
                 yolo_imgsz: int = 640,
                 yolo_device: Optional[str] = None,
                 tracker_point: str = "centroid",
                 track_stitching: bool = False,
                 track_stitching_max_frames: int = 30,
                 track_stitching_max_dist: float = 120.0):
        """
        Object Tracker utilizing YOLO and ByteTrack/BoT-SORT with a Dual-Zone State Machine,
        footprint contact tracking, and spatial-temporal track stitching.
        
        Args:
            model_name: Name of the YOLO model to use (defaults to yolov8n.pt)
            tripwire_line: Normalized tripwire line coordinates [(x1, y1), (x2, y2)]
            snapshot_dir: Directory where person crops are saved
            dead_zone_width: Width of the hysteresis threshold dead zone around the tripwire,
                as a fraction of frame height (0.0 to 1.0).
            tripwire_strict_segment: If True, restricts crossings to projection falling strictly within the segment.
            conf: Minimum detection confidence threshold for YOLO person detections (0.0 to 1.0).
            track_buffer: Number of frames to keep lost tracks alive before reassigning a new tracker ID.
            yolo_imgsz: Image size (resolution) for YOLO model inference.
            yolo_device: Target device to run YOLO model on (e.g. 'cpu', 'cuda', '0').
            tracker_point: Anchor point on bounding box: 'bottom_center' (footprint ground contact) or 'centroid'.
            track_stitching: Whether to stitch fragmented tracks across the threshold if dropped temporarily.
            track_stitching_max_frames: Maximum frame gap between dropped track and new track to stitch.
            track_stitching_max_dist: Maximum Euclidean pixel distance between dropped track and new track to stitch.
        """
        # Disable Ultralytics telemetry and sync settings to prevent periodic network lag
        try:
            from ultralytics import settings
            settings.update({
                "sync": False,
                "clearml": False,
                "comet": False,
                "dvc": False,
                "hub": False,
                "mlflow": False,
                "neptune": False,
                "raytune": False,
                "tensorboard": False,
                "wandb": False
            })
        except Exception as e:
            logger.warning(f"Could not disable YOLO telemetry/sync settings: {e}")

        # Load YOLO model
        self.model = YOLO(model_name)
        
        # Unique session ID for this instantiation
        self.session_id = str(uuid.uuid4())
        
        # Normalized tripwire line segment, e.g. [(0.2, 0.5), (0.8, 0.5)]
        self.tripwire_line = tripwire_line or [(0.2, 0.5), (0.8, 0.5)]
        self.snapshot_dir = snapshot_dir
        self.dead_zone_width = dead_zone_width
        self.tripwire_strict_segment = tripwire_strict_segment
        self.conf = conf
        self.track_buffer = track_buffer
        self.yolo_imgsz = yolo_imgsz
        self.yolo_device = yolo_device
        self.tracker_point = tracker_point.lower().strip()
        self.track_stitching = track_stitching
        self.track_stitching_max_frames = track_stitching_max_frames
        self.track_stitching_max_dist = track_stitching_max_dist
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Generate a custom tracker config with the specified track_buffer
        self._tracker_config_path = self._create_tracker_config(track_buffer, self.snapshot_dir)
        logger.info(
            f"ObjectTracker initialized: conf={self.conf}, track_buffer={track_buffer}, "
            f"dead_zone_width={dead_zone_width}, anchor={self.tracker_point}, stitching={self.track_stitching}"
        )
        
        # Track history: mapping of tracker_id -> list of anchor points (x, y)
        self.track_histories: Dict[int, List[Tuple[float, float]]] = {}
        # Track confirmed side state: mapping of tracker_id -> last known confirmed side (+1 or -1 or 0)
        self.track_confirmed_sides: Dict[int, int] = {}
        # Track zone history: mapping of tracker_id -> list of zones visited (-1, 0, 1)
        self.track_zone_history: Dict[int, List[int]] = {}
        # Track events already triggered to prevent duplicate firings per crossing
        self.track_events_triggered: Dict[int, Set[str]] = {}
        # Track last seen frame count to implement TTL buffer
        self.track_last_seen: Dict[int, int] = {}
        # Internal frame counter
        self.frame_count: int = 0
        # Expose latest processed bounding boxes for visualization
        self.latest_boxes = None
        # Recent lost tracks buffer for spatial-temporal stitching
        self._recent_lost_tracks: List[Dict[str, Any]] = []

    @staticmethod
    def _create_tracker_config(track_buffer: int, snapshot_dir: str) -> str:
        """Creates a custom BoT-SORT tracker config YAML with the specified track_buffer."""
        config_content = f"""# Custom BoT-SORT tracker config (auto-generated by ObjectTracker)
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {track_buffer}
match_thresh: 0.8
fuse_score: true
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: false
"""
        config_path = os.path.join(snapshot_dir, "tracker_config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)
        logger.debug(f"Custom tracker config written to: {config_path}")
        return config_path

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
        Result is in pixels when called from process_frame.
        """
        dx = B[0] - A[0]
        dy = B[1] - A[1]
        line_len = math.sqrt(dx * dx + dy * dy)
        if line_len < 1e-9:
            return 0.0
        return (dx * (P[1] - A[1]) - dy * (P[0] - A[0])) / line_len

    def _is_point_in_segment_bounds(self, A: Tuple[float, float], B: Tuple[float, float], P: Tuple[float, float]) -> bool:
        """
        Checks if the projection of point P onto the line defined by AB falls within the line segment AB.
        Allows a small margin (-0.1 to 1.1) to account for bounding box width at the segment edges.
        """
        dot_product = (P[0] - A[0]) * (B[0] - A[0]) + (P[1] - A[1]) * (B[1] - A[1])
        segment_len_sq = (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2
        
        if segment_len_sq < 1e-9:
            return False
            
        t = dot_product / segment_len_sq
        return -0.1 <= t <= 1.1

    def save_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], tracker_id: int, event_type: str) -> Optional[str]:
        """Saves a high-resolution crop of the person bounding box."""
        try:
            h, w, _ = frame.shape
            x1, y1, x2, y2 = bbox
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

    def process_frame(self, frame: np.ndarray) -> List[CrossingEvent]:
        """
        Runs object tracking on the frame using Dual-Zone State Machine and footprint anchor tracking.
        
        Returns:
            List of detected crossing events:
            [CrossingEvent(event_type="ENTER"/"LEAVE", tracker_id=id, confidence=conf, snapshot_path=path)]
        """
        self.frame_count += 1
        self.latest_boxes = None
        h, w, _ = frame.shape
        
        # Compute dead zone threshold in pixel space (based on frame height)
        dead_zone_half_px = (self.dead_zone_width * h) / 2.0
        
        # Convert normalized tripwire coordinates to pixel values
        tx1, ty1 = int(self.tripwire_line[0][0] * w), int(self.tripwire_line[0][1] * h)
        tx2, ty2 = int(self.tripwire_line[1][0] * w), int(self.tripwire_line[1][1] * h)
        A = (float(tx1), float(ty1))
        B = (float(tx2), float(ty2))

        # Run tracking. Classes=0 is 'person'.
        results = self.model.track(frame, persist=True, classes=[0], verbose=False,
                                    conf=self.conf, tracker=self._tracker_config_path,
                                    imgsz=self.yolo_imgsz, device=self.yolo_device)
        
        events = []
        active_tracker_ids = set()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            self.latest_boxes = boxes
            
            if boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                tracker_ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for i, tracker_id in enumerate(tracker_ids):
                    active_tracker_ids.add(tracker_id)
                    self.track_last_seen[tracker_id] = self.frame_count
                    x1, y1, x2, y2 = xyxy[i]
                    conf = confidences[i]
                    
                    # Compute tracking anchor point: bottom_center (foot contact) or centroid
                    if self.tracker_point == "bottom_center":
                        curr_point = (float((x1 + x2) / 2.0), float(y2))
                    else:
                        curr_point = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
                    
                    signed_dist = self._get_signed_distance(A, B, curr_point)
                    
                    # Evaluate current spatial zone
                    if signed_dist > dead_zone_half_px:
                        current_zone = 1  # Inside (+1)
                    elif signed_dist < -dead_zone_half_px:
                        current_zone = -1  # Outside (-1)
                    else:
                        current_zone = 0  # Threshold / Dead Zone (0)
                    
                    # Initialize track history & check for spatial-temporal track stitching
                    if tracker_id not in self.track_histories:
                        self.track_histories[tracker_id] = []
                        self.track_zone_history[tracker_id] = []
                        self.track_events_triggered[tracker_id] = set()

                        stitched_from = None
                        stitch_dist = 0.0
                        stitch_gap = 0
                        if self.track_stitching:
                            # 1. First check currently inactive tracks in active track histories (dropped recently before track_buffer expiry)
                            for old_tid in list(self.track_histories.keys()):
                                if old_tid not in active_tracker_ids and old_tid in self.track_last_seen:
                                    gap = self.frame_count - self.track_last_seen[old_tid]
                                    if 1 <= gap <= self.track_stitching_max_frames:
                                        old_pt = self.track_histories[old_tid][-1] if self.track_histories[old_tid] else curr_point
                                        d = math.dist(curr_point, old_pt)
                                        if d <= self.track_stitching_max_dist:
                                            stitched_from = {
                                                "tracker_id": old_tid,
                                                "confirmed_side": self.track_confirmed_sides.get(old_tid, 0),
                                                "zone_history": self.track_zone_history.get(old_tid, []),
                                                "events_triggered": self.track_events_triggered.get(old_tid, set()),
                                                "last_frame": self.track_last_seen[old_tid]
                                            }
                                            stitch_dist = d
                                            stitch_gap = gap
                                            # Clean up old inactive track to prevent ghost reactivation
                                            del self.track_histories[old_tid]
                                            self.track_confirmed_sides.pop(old_tid, None)
                                            self.track_zone_history.pop(old_tid, None)
                                            self.track_events_triggered.pop(old_tid, None)
                                            self.track_last_seen.pop(old_tid, None)
                                            break

                            # 2. Check purged lost tracks buffer
                            if stitched_from is None and self._recent_lost_tracks:
                                for idx, lost in enumerate(self._recent_lost_tracks):
                                    frame_gap = self.frame_count - lost["last_frame"]
                                    dist = math.dist(curr_point, lost["last_point"])
                                    max_allowed_gap = self.track_buffer + self.track_stitching_max_frames
                                    if frame_gap <= max_allowed_gap and dist <= self.track_stitching_max_dist:
                                        stitched_from = lost
                                        stitch_dist = dist
                                        stitch_gap = frame_gap
                                        self._recent_lost_tracks.pop(idx)
                                        break
                                    
                        if stitched_from is not None:
                            self.track_confirmed_sides[tracker_id] = stitched_from["confirmed_side"]
                            self.track_zone_history[tracker_id] = list(stitched_from["zone_history"])
                            self.track_events_triggered[tracker_id] = set(stitched_from["events_triggered"])
                            logger.info(
                                f"Stitched new tracker {tracker_id} at ({curr_point[0]:.1f}, {curr_point[1]:.1f}) "
                                f"to dropped tracker {stitched_from['tracker_id']} (origin: {stitched_from['confirmed_side']}, "
                                f"gap: {stitch_gap} frames, dist: {stitch_dist:.1f}px)"
                            )
                        else:
                            if current_zone != 0:
                                self.track_confirmed_sides[tracker_id] = current_zone
                            else:
                                self.track_confirmed_sides[tracker_id] = self._get_point_side(A, B, curr_point)
                            side_label = {1: "INSIDE (+1)", -1: "OUTSIDE (-1)", 0: "COLLINEAR (0)"}.get(
                                self.track_confirmed_sides[tracker_id], str(self.track_confirmed_sides[tracker_id])
                            )
                            logger.info(f"New tracker ID {tracker_id} created at ({curr_point[0]:.1f}, {curr_point[1]:.1f}), anchor: {self.tracker_point}, initial side: {side_label}, conf: {conf:.2f}")
                    
                    # Update zone history
                    if not self.track_zone_history[tracker_id] or self.track_zone_history[tracker_id][-1] != current_zone:
                        self.track_zone_history[tracker_id].append(current_zone)

                    confirmed_side = self.track_confirmed_sides[tracker_id]
                    event_type = None
                    
                    in_bounds = True
                    if self.tripwire_strict_segment:
                        in_bounds = self._is_point_in_segment_bounds(A, B, curr_point)
                    
                    # Dual-Zone State Machine transitions:
                    # OUTSIDE (-1) -> INSIDE (+1) registers ENTER
                    # INSIDE (+1) -> OUTSIDE (-1) registers LEAVE
                    if confirmed_side == -1 and current_zone == 1:
                        self.track_confirmed_sides[tracker_id] = 1
                        if in_bounds and "ENTER" not in self.track_events_triggered[tracker_id]:
                            event_type = "ENTER"
                            self.track_events_triggered[tracker_id].add("ENTER")
                            logger.info(f"Tracker {tracker_id} crossed dual-zone: OUTSIDE (-1) -> INSIDE (+1) (ENTER), signed_dist: {signed_dist:.1f}, dead_zone_half: {dead_zone_half_px:.1f}")
                    elif confirmed_side == 1 and current_zone == -1:
                        self.track_confirmed_sides[tracker_id] = -1
                        if in_bounds and "LEAVE" not in self.track_events_triggered[tracker_id]:
                            event_type = "LEAVE"
                            self.track_events_triggered[tracker_id].add("LEAVE")
                            logger.info(f"Tracker {tracker_id} crossed dual-zone: INSIDE (+1) -> OUTSIDE (-1) (LEAVE), signed_dist: {signed_dist:.1f}, dead_zone_half: {dead_zone_half_px:.1f}")
                    elif confirmed_side == 0:
                        if current_zone != 0:
                            self.track_confirmed_sides[tracker_id] = current_zone
                            side_label = "INSIDE (+1)" if current_zone == 1 else "OUTSIDE (-1)"
                            logger.info(f"Tracker {tracker_id} resolved from THRESHOLD (0) -> {side_label}, signed_dist: {signed_dist:.1f}")

                    if event_type:
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        snapshot_path = self.save_crop(frame, bbox, tracker_id, event_type)
                        events.append(CrossingEvent(
                            event_type=event_type,
                            tracker_id=int(tracker_id),
                            confidence=float(conf),
                            snapshot_path=snapshot_path
                        ))
                                
                    # Append current point to history and limit size to last 10 points
                    self.track_histories[tracker_id].append(curr_point)
                    if len(self.track_histories[tracker_id]) > 10:
                        self.track_histories[tracker_id].pop(0)

        # Cleanup old tracking histories that are no longer active to prevent memory leaks
        dead_ids = []
        for tid in self.track_histories:
            last_seen = self.track_last_seen.get(tid, 0)
            if tid not in active_tracker_ids and (self.frame_count - last_seen) > self.track_buffer:
                dead_ids.append(tid)
                
        for tid in dead_ids:
            last_side = self.track_confirmed_sides.get(tid, 0)
            last_pt = self.track_histories[tid][-1] if self.track_histories[tid] else (0.0, 0.0)
            side_label = {1: "INSIDE", -1: "OUTSIDE", 0: "THRESHOLD"}.get(last_side, str(last_side))
            logger.info(f"Tracker {tid} lost (inactive >{self.track_buffer} frames). Last confirmed side: {side_label}. Adding to stitching buffer.")
            
            # Store in recent lost buffer for stitching
            if self.track_stitching:
                self._recent_lost_tracks.append({
                    "tracker_id": tid,
                    "last_frame": self.track_last_seen.get(tid, self.frame_count),
                    "dropped_at_frame": self.frame_count,
                    "last_point": last_pt,
                    "confirmed_side": last_side,
                    "zone_history": list(self.track_zone_history.get(tid, [])),
                    "events_triggered": set(self.track_events_triggered.get(tid, set()))
                })
            
            del self.track_histories[tid]
            if tid in self.track_confirmed_sides:
                del self.track_confirmed_sides[tid]
            if tid in self.track_zone_history:
                del self.track_zone_history[tid]
            if tid in self.track_events_triggered:
                del self.track_events_triggered[tid]
            if tid in self.track_last_seen:
                del self.track_last_seen[tid]

        # Prune stale lost tracks beyond track_buffer + 2x stitching window
        if self._recent_lost_tracks:
            cutoff = self.frame_count - (self.track_buffer + self.track_stitching_max_frames * 2)
            self._recent_lost_tracks = [t for t in self._recent_lost_tracks if t["last_frame"] >= cutoff]

        return events

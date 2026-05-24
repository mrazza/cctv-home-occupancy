import cv2
import time
import threading
from typing import Optional, List, Dict, Any, Tuple
from src.config import CONFIG
from src.motion_detector import MotionDetector
from src.object_tracker import ObjectTracker
from src.database import DatabaseManager

class FrameRegistry:
    _lock = threading.Lock()
    _last_frame: Optional[cv2.Mat] = None
    _last_timestamp: float = 0.0

    @classmethod
    def set_frame(cls, frame: cv2.Mat):
        """Thread-safe update of the latest frame."""
        with cls._lock:
            cls._last_frame = frame.copy()
            cls._last_timestamp = time.time()

    @classmethod
    def get_frame(cls) -> Optional[cv2.Mat]:
        """Thread-safe retrieval of the latest frame."""
        with cls._lock:
            if cls._last_frame is not None:
                return cls._last_frame.copy()
            return None


class ThreadedVideoReader:
    def __init__(self, src: str):
        """Thread-safe background video reader to always retrieve the freshest frame."""
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame: Optional[cv2.Mat] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
                else:
                    # If stream fails, wait before retry
                    time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


class PipelineOrchestrator:
    def __init__(self, 
                 db_manager: DatabaseManager,
                 motion_detector: Optional[MotionDetector] = None,
                 object_tracker: Optional[ObjectTracker] = None,
                 cooldown_frames: int = 150,
                 fps_limit: int = 10):
        """
        Orchestrates the "Fast & Slow" pipeline switching between low-CPU motion detection and high-accuracy YOLO.
        """
        self.db = db_manager
        self.motion_detector = motion_detector or MotionDetector()
        self.object_tracker = object_tracker or ObjectTracker()
        self.cooldown_frames = cooldown_frames
        self.fps_limit = fps_limit
        
        # State
        self.state = "IDLE"  # "IDLE" or "ACTIVE"
        self.cooldown_counter = 0
        self.running = False

    def process_single_frame(self, frame: cv2.Mat) -> List[Dict[str, Any]]:
        """
        Processes a single frame through the pipeline and updates the database state accordingly.
        Returns crossing events captured in this frame.
        """
        # Always run motion detection to keep background model updated
        motion_detected = self.motion_detector.detect(frame)
        
        events = []
        
        if self.state == "IDLE":
            if motion_detected:
                self.state = "ACTIVE"
                self.cooldown_counter = self.cooldown_frames
                # Trigger YOLO tracker on this frame
                events = self.object_tracker.process_frame(frame)
        
        elif self.state == "ACTIVE":
            # Run YOLO Tracker
            events = self.object_tracker.process_frame(frame)
            
            # Check if we should remain active
            has_active_tracks = len(self.object_tracker.track_histories) > 0
            
            if motion_detected or has_active_tracks:
                # Reset cooldown
                self.cooldown_counter = self.cooldown_frames
            else:
                self.cooldown_counter -= 1
                if self.cooldown_counter <= 0:
                    self.state = "IDLE"
                    self.motion_detector.reset() # Reset background to adapt to any slow lighting changes
                    
        # Log detected events to database
        for event in events:
            self.db.log_event(
                event_type=event["event_type"],
                tracker_id=event["tracker_id"],
                confidence=event["confidence"],
                snapshot_path=event["snapshot_path"]
            )
            
        return events

    def run_on_stream(self, rtsp_url: str):
        """Runs the monitoring pipeline continuously on an RTSP stream (Blocking)."""
        reader = ThreadedVideoReader(rtsp_url).start()
        self.running = True
        
        delay = 1.0 / self.fps_limit
        
        try:
            while self.running:
                start_time = time.time()
                ret, frame = reader.read()
                
                if ret and frame is not None:
                    # Update the global frame registry for API/external access
                    FrameRegistry.set_frame(frame)
                    # Process frame
                    self.process_single_frame(frame)
                
                # Regulate Frame Rate
                elapsed = time.time() - start_time
                sleep_time = max(0.0, delay - elapsed)
                time.sleep(sleep_time)
        finally:
            reader.stop()

    def stop(self):
        self.running = False

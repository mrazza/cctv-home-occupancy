import cv2
import time
import threading
import logging
from typing import Optional, List, Dict, Any, Tuple
from src.types import CrossingEvent
from src.config import CONFIG
from src.motion_detector import MotionDetector
from src.object_tracker import ObjectTracker
from src.database import DatabaseManager

logger = logging.getLogger(__name__)

class FrameRegistry:
    _lock = threading.Lock()
    _last_frame: Optional[cv2.Mat] = None
    _last_timestamp: float = 0.0
    _last_timestamp_iso: Optional[str] = None

    @classmethod
    def set_frame(cls, frame: cv2.Mat):
        """Thread-safe update of the latest frame."""
        from datetime import datetime
        with cls._lock:
            cls._last_frame = frame.copy()
            cls._last_timestamp = time.time()
            cls._last_timestamp_iso = datetime.now().isoformat()

    @classmethod
    def get_frame(cls) -> Optional[cv2.Mat]:
        """Thread-safe retrieval of the latest frame."""
        with cls._lock:
            if cls._last_frame is not None:
                return cls._last_frame.copy()
            return None

    @classmethod
    def get_last_timestamp_iso(cls) -> Optional[str]:
        """Thread-safe retrieval of the last processed frame ISO timestamp."""
        with cls._lock:
            return cls._last_timestamp_iso


class OrchestratorRegistry:
    _instance: Optional['PipelineOrchestrator'] = None

    @classmethod
    def set_instance(cls, instance: 'PipelineOrchestrator'):
        cls._instance = instance

    @classmethod
    def get_instance(cls) -> Optional['PipelineOrchestrator']:
        return cls._instance


class ThreadedVideoReader:
    def __init__(self, src: str, buffer_size: Optional[int] = None):
        """Thread-safe background video reader to always retrieve the freshest frame."""
        self.src = src
        self.cap = cv2.VideoCapture(src)
        
        # Configure buffer size
        buf_size = buffer_size if buffer_size is not None else CONFIG.video_buffer_size
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)
        
        self.ret = False
        self.frame: Optional[cv2.Mat] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self._last_read_time = time.time()
        self._frame_count = 0

    def start(self):
        if self.running:
            return self
        logger.info(f"Starting ThreadedVideoReader on source: {self.src}")
        self.running = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        return self

    def _update(self):
        logger.info("ThreadedVideoReader update loop started.")
        while self.running:
            if not self.cap.isOpened():
                logger.warning(f"RTSP stream connection lost or not opened. Attempting to reconnect to source: {self.src}")
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.src)
                self._last_read_time = time.time()
                continue

            try:
                t0 = time.time()
                ret, frame = self.cap.read()
                elapsed = time.time() - t0
                if elapsed > 0.1:
                    logger.warning(f"ThreadedVideoReader: cap.read() took {elapsed * 1000.0:.1f}ms")
            except Exception:
                logger.exception("Exception occurred during cv2.VideoCapture.read()")
                ret = False
                frame = None

            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame
                self._last_read_time = time.time()
                self._frame_count += 1
                if self._frame_count % 3 == 0:  # Use a lower divisor so it's easily coverable in tests
                    logger.debug(f"Heartbeat: ThreadedVideoReader has successfully processed {self._frame_count} frames.")
            else:
                # If we haven't successfully read a frame for more than 10.0 seconds, trigger reconnect
                if time.time() - self._last_read_time > 10.0:
                    logger.warning(f"RTSP stream read timeout (> 10s) or read failure. Reconnecting to source: {self.src}")
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = cv2.VideoCapture(self.src)
                    self._last_read_time = time.time()  # Reset to prevent continuous instant reconnect loops
                else:
                    time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        with self.lock:
            ret = self.ret
            self.ret = False  # Mark as consumed so subsequent reads return False until a new frame is retrieved
            return ret, self.frame

    def stop(self):
        logger.info("Stopping ThreadedVideoReader...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        logger.info("ThreadedVideoReader stopped.")


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
        self.motion_detector = motion_detector or MotionDetector(
            threshold=CONFIG.motion_threshold,
            min_contour_area=CONFIG.motion_min_contour_area,
            alpha=CONFIG.background_alpha,
            roi=CONFIG.motion_roi
        )
        self.object_tracker = object_tracker or ObjectTracker(
            model_name=CONFIG.model_name,
            tripwire_line=CONFIG.tripwire_line,
            snapshot_dir=CONFIG.snapshot_dir,
            dead_zone_width=CONFIG.tripwire_dead_zone_width,
            tripwire_strict_segment=CONFIG.tripwire_strict_segment,
            conf=CONFIG.tracker_confidence,
            track_buffer=CONFIG.track_buffer,
            yolo_imgsz=CONFIG.yolo_imgsz,
            yolo_device=CONFIG.yolo_device
        )
        self.cooldown_frames = cooldown_frames
        self.fps_limit = fps_limit
        
        # State
        self.state = "IDLE"  # "IDLE" or "ACTIVE"
        self.cooldown_counter = 0
        self.running = False
        
        # On-Demand event-driven settings
        self.active_until = 0.0
        self.lock = threading.Lock()
        self.reader = None
        self.rtsp_url = None
        
        OrchestratorRegistry.set_instance(self)
        logger.info(f"PipelineOrchestrator initialized. Cooldown frames: {cooldown_frames}, FPS limit: {fps_limit}")

    def trigger_event_window(self, duration: int):
        """Triggers or extends the active window for on-demand stream processing."""
        with self.lock:
            self.active_until = time.time() + duration
            logger.info(f"On-demand event-driven window triggered/extended until epoch {self.active_until} (for {duration}s)")

    def process_single_frame(self, frame: cv2.Mat) -> List[CrossingEvent]:
        """
        Processes a single frame through the pipeline and updates the database state accordingly.
        Returns crossing events captured in this frame.
        """
        try:
            # Always run motion detection to keep background model updated
            motion_detected = self.motion_detector.detect(frame)
            
            events = []
            
            if self.state == "IDLE":
                if motion_detected:
                    logger.info("Motion detected! Pipeline transitioning from IDLE to ACTIVE. Triggering YOLO tracker.")
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
                    if self.cooldown_counter != self.cooldown_frames:
                        logger.debug(f"Resetting cooldown counter. (Active tracks: {has_active_tracks})")
                    self.cooldown_counter = self.cooldown_frames
                else:
                    self.cooldown_counter -= 1
                    logger.debug(f"No motion or active tracks. Cooldown counter decremented to: {self.cooldown_counter}")
                    if self.cooldown_counter <= 0:
                        logger.info("No motion or active tracks detected within cooldown window. Pipeline reverting to IDLE.")
                        self.state = "IDLE"
                        self.motion_detector.reset() # Reset background to adapt to any slow lighting changes
                        
            # Log detected events to database
            for event in events:
                logger.info(f"Tripwire crossing event detected: {event.event_type} (Tracker ID: {event.tracker_id}, Confidence: {event.confidence:.2f})")
                event_id = self.db.log_event(
                    event_type=event.event_type,
                    tracker_id=event.tracker_id,
                    confidence=event.confidence,
                    snapshot_path=event.snapshot_path,
                    session_id=self.object_tracker.session_id
                )
                logger.info(f"Successfully logged crossing event to DB (Event ID: {event_id})")
                
                # Check if we should trigger webhooks
                if CONFIG.webhook_urls:
                    try:
                        state = self.db.get_current_state()
                        payload = {
                            "event_id": event_id,
                            "event_type": event.event_type,
                            "tracker_id": event.tracker_id,
                            "confidence": event.confidence,
                            "snapshot_path": event.snapshot_path,
                            "is_someone_home": state.is_someone_home,
                            "current_occupancy": state.current_occupancy,
                            "timestamp": state.last_updated
                        }
                    except Exception as db_err:
                        logger.error(f"Failed to fetch current state for webhook payload: {db_err}")
                        payload = {
                            "event_id": event_id,
                            "event_type": event.event_type,
                            "tracker_id": event.tracker_id,
                            "confidence": event.confidence,
                            "snapshot_path": event.snapshot_path,
                            "is_someone_home": None,
                            "current_occupancy": None,
                            "timestamp": None
                        }
                    
                    for url in CONFIG.webhook_urls:
                        threading.Thread(
                            target=self._dispatch_webhook_thread,
                            args=(url, payload),
                            daemon=True
                        ).start()
                
            return events
        except Exception:
            logger.exception("Unexpected error occurred while processing a single frame in the pipeline.")
            return []

    def run_on_stream(self, rtsp_url: str):
        """Runs the monitoring pipeline continuously on an RTSP stream (Blocking)."""
        self.rtsp_url = rtsp_url
        logger.info(f"Initializing stream reader with trigger mode: '{CONFIG.trigger_mode}' for: {rtsp_url}")
        
        # Start continuous reader immediately if in continuous mode
        if CONFIG.trigger_mode == "continuous":
            self.reader = ThreadedVideoReader(rtsp_url).start()
            
        self.running = True
        delay = 1.0 / self.fps_limit
        logger.info(f"Stream pipeline started with delay of {delay:.3f}s between checks.")
        
        try:
            while self.running:
                start_time = time.time()
                
                is_active = True
                if CONFIG.trigger_mode == "event":
                    # Determine if trigger window is active
                    if time.time() < self.active_until:
                        if self.reader is None or not self.reader.running:
                            logger.info("Trigger window active. Lazily starting ThreadedVideoReader...")
                            self.reader = ThreadedVideoReader(rtsp_url).start()
                    else:
                        is_active = False
                        # Trigger window expired. Check if YOLO is still actively tracking something
                        has_active_tracks = len(self.object_tracker.track_histories) > 0
                        if has_active_tracks:
                            # Keep tracking until they cross or leave
                            with self.lock:
                                self.active_until = time.time() + 5.0  # Extend slightly
                            is_active = True
                            if self.reader is None or not self.reader.running:
                                logger.info("Active tracks found. Lazily starting ThreadedVideoReader...")
                                self.reader = ThreadedVideoReader(rtsp_url).start()
                        elif self.reader is not None:
                            logger.info("Trigger window expired and no active tracks. Stopping ThreadedVideoReader to release RTSP quota...")
                            self.reader.stop()
                            self.reader = None
                            self.state = "IDLE"
                            
                if is_active and self.reader is not None:
                    ret, frame = self.reader.read()
                    if ret and frame is not None:
                        # Update the global frame registry for API/external access
                        FrameRegistry.set_frame(frame)
                        # Process frame
                        self.process_single_frame(frame)
                
                # Regulate Frame Rate
                elapsed = time.time() - start_time
                sleep_time = max(0.0, delay - elapsed)
                time.sleep(sleep_time)
        except Exception:
            self.running = False
            logger.exception("Continuous stream processing pipeline crashed due to an unhandled exception.")
        finally:
            logger.info("Stopping continuous stream processing pipeline...")
            if self.reader is not None:
                self.reader.stop()
                self.reader = None
            logger.info("Continuous stream processing pipeline stopped.")

    def _dispatch_webhook_thread(self, url: str, payload: Dict[str, Any]):
        """Sends the HTTP POST request to the webhook URL in a background thread."""
        import httpx
        try:
            logger.info(f"Dispatching event webhook to: {url}")
            response = httpx.post(url, json=payload, timeout=CONFIG.webhook_timeout)
            if response.status_code >= 400:
                logger.warning(f"Webhook {url} returned non-success status code: {response.status_code}")
            else:
                logger.info(f"Webhook {url} successfully triggered.")
        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        except Exception as exc:
            logger.exception(f"Unexpected error dispatching webhook {url}: {exc}")

    def stop(self):
        logger.info("Orchestrator stop requested.")
        self.running = False

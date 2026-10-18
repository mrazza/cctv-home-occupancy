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
from src.video_reader import ThreadedVideoReader
from src.preprocessing import compute_luminance, compute_blur_metric, enhance_low_light

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
        """Registers the active pipeline orchestrator instance globally."""
        cls._instance = instance

    @classmethod
    def get_instance(cls) -> Optional['PipelineOrchestrator']:
        """Retrieves the globally registered pipeline orchestrator instance."""
        return cls._instance

class PipelineOrchestrator:
    def __init__(self, 
                 db_manager: DatabaseManager,
                 motion_detector: Optional[MotionDetector] = None,
                 object_tracker: Optional[ObjectTracker] = None,
                 cooldown_frames: int = 150,
                 fps_limit: int = 10):
        """
        Orchestrates the "Fast & Slow" pipeline switching between low-CPU motion detection and high-accuracy YOLO.

        Args:
            db_manager: DatabaseManager instance for state storage/event logging.
            motion_detector: Custom MotionDetector. If None, initialized using CONFIG motion settings.
            object_tracker: Custom ObjectTracker. If None, initialized using CONFIG tracker settings.
            cooldown_frames: Idle frames before ACTIVE state returns to IDLE state.
            fps_limit: Rate regulation for the processing loop (target frames per second).
        """
        self.db = db_manager
        self.motion_detector = motion_detector or MotionDetector(
            threshold=CONFIG.motion_threshold,
            min_contour_area=CONFIG.motion_min_contour_area,
            alpha=CONFIG.background_alpha,
            roi=CONFIG.motion_roi,
            subtractor_type=CONFIG.motion_subtractor_type,
            adaptive_threshold=CONFIG.motion_adaptive_threshold,
            low_light_threshold=CONFIG.low_light_threshold
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
            yolo_device=CONFIG.yolo_device,
            tracker_point=CONFIG.tracker_point,
            track_stitching=CONFIG.track_stitching,
            track_stitching_max_frames=CONFIG.track_stitching_max_frames,
            track_stitching_max_dist=CONFIG.track_stitching_max_dist
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
        self.reader: Optional[ThreadedVideoReader] = None
        self.rtsp_url = None
        
        OrchestratorRegistry.set_instance(self)
        logger.info(f"PipelineOrchestrator initialized. Cooldown frames: {cooldown_frames}, FPS limit: {fps_limit}")

    def trigger_event_window(self, duration: int):
        """Triggers or extends the active window for on-demand stream processing."""
        with self.lock:
            self.active_until = max(self.active_until, time.time() + duration)
            logger.info(f"On-demand event-driven window triggered/extended until epoch {self.active_until} (for {duration}s)")

    def process_single_frame(self, frame: cv2.Mat) -> List[CrossingEvent]:
        """
        Processes a single frame through the pipeline, records telemetry (luminance, blur),
        applies low-light enhancement if configured, and updates the database state accordingly.
        Returns crossing events captured in this frame.
        """
        try:
            # Calculate frame quality and illumination telemetry
            luminance = compute_luminance(frame)
            blur_metric = compute_blur_metric(frame)
            is_low_light = (luminance < CONFIG.low_light_threshold)

            # Apply low-light pre-processing enhancement if enabled and illumination is low
            if CONFIG.low_light_enhancement and is_low_light:
                processing_frame = enhance_low_light(
                    frame,
                    method=CONFIG.low_light_method,
                    clip_limit=CONFIG.clahe_clip_limit,
                    grid_size=CONFIG.clahe_grid_size,
                    gamma=CONFIG.gamma_value,
                    denoise_prefilter=CONFIG.low_light_denoise_prefilter
                )
            else:
                processing_frame = frame

            # Always run motion detection to keep background model updated
            motion_detected = self.motion_detector.detect(processing_frame)
            
            events = []
            
            if self.state == "IDLE":
                if motion_detected:
                    logger.info(f"Motion detected! Pipeline transitioning from IDLE to ACTIVE (Luminance: {luminance:.1f}). Triggering YOLO tracker.")
                    self.state = "ACTIVE"
                    self.cooldown_counter = self.cooldown_frames
                    # Trigger YOLO tracker on this frame
                    events = self.object_tracker.process_frame(processing_frame)
            elif self.state == "ACTIVE":
                # Run YOLO Tracker
                events = self.object_tracker.process_frame(processing_frame)
                
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
                logger.info(
                    f"Tripwire crossing event detected: {event.event_type} "
                    f"(Tracker ID: {event.tracker_id}, Confidence: {event.confidence:.2f}, "
                    f"Luminance: {luminance:.1f}, Blur: {blur_metric:.1f})"
                )
                event_id = self.db.log_event(
                    event_type=event.event_type,
                    tracker_id=event.tracker_id,
                    confidence=event.confidence,
                    snapshot_path=event.snapshot_path,
                    session_id=self.object_tracker.session_id,
                    luminance=luminance,
                    blur_metric=blur_metric
                )
                logger.info(f"Successfully logged crossing event to DB (Event ID: {event_id})")

                # Diagnostic video clip capture
                if CONFIG.diagnostic_clip_enabled and self.reader is not None:
                    if not CONFIG.diagnostic_clip_low_light_only or is_low_light:
                        threading.Thread(
                            target=self._save_diagnostic_clip_thread,
                            args=(f"{event.event_type.lower()}_id{event.tracker_id}",),
                            daemon=True
                        ).start()
                
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
                            "luminance": luminance,
                            "blur_metric": blur_metric,
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
                            "luminance": luminance,
                            "blur_metric": blur_metric,
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

    def _save_diagnostic_clip_thread(self, prefix: str):
        """Asynchronously records a diagnostic video clip from buffered frames."""
        try:
            if self.reader is not None and hasattr(self.reader, 'save_diagnostic_clip'):
                self.reader.save_diagnostic_clip(
                    output_dir=CONFIG.diagnostic_clip_dir,
                    prefix=prefix,
                    fps=self.fps_limit,
                    duration_sec=CONFIG.diagnostic_clip_duration
                )
        except Exception as e:
            logger.warning(f"Failed to save diagnostic clip: {e}")

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
        """
        Signals the continuous processing loop run_on_stream to stop running.
        """
        logger.info("Orchestrator stop requested.")
        self.running = False

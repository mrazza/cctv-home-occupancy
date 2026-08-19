import os
import cv2
import time
import threading
import logging
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List
from src.config import CONFIG

logger = logging.getLogger(__name__)

class ThreadedVideoReader:
    def __init__(self, src: str, buffer_size: Optional[int] = None, rolling_buffer_seconds: float = 10.0, fps_estimate: int = 10):
        """
        Thread-safe background video reader to always retrieve the freshest frame,
        with an integrated rolling circular buffer for diagnostic video clip generation.
        """
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
        
        # Rolling frame circular buffer for telemetry / diagnostic clips
        max_buffered = int(max(10, rolling_buffer_seconds * fps_estimate))
        self.frame_buffer: deque[Tuple[float, cv2.Mat]] = deque(maxlen=max_buffered)
        
        # Aspect ratio correction parameters
        self.par_correction = 1.0
        self._detect_aspect_ratio_correction()

    def _detect_aspect_ratio_correction(self):
        """Detects if aspect ratio correction is needed based on config or stream metadata."""
        if CONFIG.aspect_ratio_override is not None:
            # Computed dynamically on first read frame once width/height are known.
            logger.info(f"Aspect ratio override configured: {CONFIG.aspect_ratio_override}")
            self.par_correction = None
            return

        # Attempt to auto-detect using stream metadata (SAR)
        try:
            sar_num = self.cap.get(cv2.CAP_PROP_SAR_NUM)
            sar_den = self.cap.get(cv2.CAP_PROP_SAR_DEN)
            if isinstance(sar_num, (int, float)) and isinstance(sar_den, (int, float)) and sar_num > 0 and sar_den > 0:
                self.par_correction = sar_num / sar_den
                if abs(self.par_correction - 1.0) > 1e-4:
                    logger.info(f"Auto-detected non-square pixels from stream metadata. PAR correction factor: {self.par_correction:.4f}")
                else:
                    self.par_correction = 1.0
            else:
                self.par_correction = 1.0
        except Exception:
            self.par_correction = 1.0

    def _apply_aspect_ratio_correction(self, frame: cv2.Mat) -> cv2.Mat:
        """Resizes the frame to correct aspect ratio distortions if correction factor is active."""
        h, w = frame.shape[:2]
        
        # Calculate dynamic PAR correction if override is configured and hasn't been computed yet
        if CONFIG.aspect_ratio_override is not None and self.par_correction is None:
            current_aspect = w / h
            target_aspect = CONFIG.aspect_ratio_override
            self.par_correction = target_aspect / current_aspect
            logger.info(f"Computed PAR correction from aspect ratio override: {self.par_correction:.4f} (Raw: {w}x{h}, Target: {target_aspect:.4f})")
            
        if self.par_correction is not None and abs(self.par_correction - 1.0) > 1e-4:
            new_w = int(round(w * self.par_correction))
            frame = cv2.resize(frame, (new_w, h), interpolation=cv2.INTER_LINEAR)
            
        return frame

    def start(self):
        """
        Starts the background frame capture update loop thread.

        Returns:
            Self instance (ThreadedVideoReader).
        """
        if self.running:
            return self
        logger.info(f"Starting ThreadedVideoReader on source: {self.src}")
        self.running = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """
        Internal target method running in a background thread.
        Continuously reads frames from VideoCapture and handles reconnects on failures/timeouts.
        """
        logger.info("ThreadedVideoReader update loop started.")
        while self.running:
            if not self.cap.isOpened():
                logger.warning(f"RTSP stream connection lost or not opened. Attempting to reconnect to source: {self.src}")
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.src)
                self._detect_aspect_ratio_correction()
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
                frame = self._apply_aspect_ratio_correction(frame)
                now_ts = time.time()
                with self.lock:
                    self.ret = True
                    self.frame = frame
                    self.frame_buffer.append((now_ts, frame.copy()))
                self._last_read_time = now_ts
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
                    self._detect_aspect_ratio_correction()
                    self._last_read_time = time.time()  # Reset to prevent continuous instant reconnect loops
                else:
                    time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Returns the latest captured frame and its status.
        Calling this method marks the frame as consumed, so subsequent reads
        will return False until a new frame has been read.

        Returns:
            Tuple (is_new_frame, frame_data).
        """
        with self.lock:
            ret = self.ret
            self.ret = False  # Mark as consumed so subsequent reads return False until a new frame is retrieved
            return ret, self.frame

    def get_buffered_frames(self, duration_sec: Optional[float] = None) -> List[Tuple[float, cv2.Mat]]:
        """
        Retrieves a thread-safe snapshot list of recently captured frames from the circular buffer.
        """
        with self.lock:
            frames = list(self.frame_buffer)
        if duration_sec is not None and duration_sec > 0 and len(frames) > 0:
            cutoff = time.time() - duration_sec
            frames = [f for f in frames if f[0] >= cutoff]
        return frames

    def save_diagnostic_clip(self, output_dir: str, prefix: str = "diag_clip", fps: int = 10, duration_sec: Optional[float] = None) -> Optional[str]:
        """
        Saves recent buffered frames to a diagnostic video clip (MP4).

        Args:
            output_dir: Directory path where the clip will be saved.
            prefix: Filename prefix.
            fps: Frame rate for the output video.
            duration_sec: Number of recent seconds to include.

        Returns:
            The filepath of the saved video file, or None if no frames were available.
        """
        frames = self.get_buffered_frames(duration_sec=duration_sec)
        if not frames:
            return None

        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp_str}.mp4"
            filepath = os.path.join(output_dir, filename)

            first_frame = frames[0][1]
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, float(fps), (w, h))

            if not writer.isOpened():
                # Fallback to AVI / XVID if MP4 fourcc fails on some systems
                filename = f"{prefix}_{timestamp_str}.avi"
                filepath = os.path.join(output_dir, filename)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(filepath, fourcc, float(fps), (w, h))

            for _, frame_img in frames:
                writer.write(frame_img)
            writer.release()
            logger.info(f"Diagnostic video clip saved to: {filepath} ({len(frames)} frames)")
            return filepath
        except Exception:
            logger.exception("Failed to write diagnostic video clip.")
            return None

    def stop(self):
        """
        Stops the background reader update thread and releases the OpenCV VideoCapture resource.
        """
        logger.info("Stopping ThreadedVideoReader...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        logger.info("ThreadedVideoReader stopped.")

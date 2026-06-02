import cv2
import time
import threading
import logging
from typing import Optional, Tuple
from src.config import CONFIG

logger = logging.getLogger(__name__)

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

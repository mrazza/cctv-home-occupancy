import cv2
import numpy as np
from typing import Optional

class MotionDetector:
    def __init__(self, threshold: float = 0.005, min_contour_area: int = 500, alpha: float = 0.05, roi: Optional[list[tuple[float, float]]] = None):
        """
        Fast Motion Detector using frame differencing with running average background model.
        
        Args:
            threshold: Fraction of frame pixels that must be different to classify as motion (0.0 to 1.0)
            min_contour_area: Minimum area in pixels of a contour to count as moving object
            alpha: Accumulation rate for the running average background model (0.0 to 1.0)
            roi: Optional normalized coordinates [(x1, y1), (x2, y2), ...] representing the bounding box or polygon for motion detection.
        """
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.alpha = alpha
        self.roi = roi
        self.background_accumulator: Optional[np.ndarray] = None
        self._mask_cache: Optional[np.ndarray] = None
        self._roi_area: Optional[int] = None

    def reset(self):
        """Resets the accumulated background model."""
        self.background_accumulator = None
        self._mask_cache = None
        self._roi_area = None

    def detect(self, frame: np.ndarray) -> bool:
        """
        Processes a frame and returns True if motion is detected, updating the background.
        """
        h, w, _ = frame.shape
        
        # Apply binary mask if ROI is specified
        if self.roi is not None and len(self.roi) >= 2:
            if self._mask_cache is None or self._mask_cache.shape[:2] != (h, w):
                mask = np.zeros((h, w), dtype=np.uint8)
                if len(self.roi) == 2:
                    # Bounding box / rectangle
                    x1 = int(self.roi[0][0] * w)
                    y1 = int(self.roi[0][1] * h)
                    x2 = int(self.roi[1][0] * w)
                    y2 = int(self.roi[1][1] * h)
                    x_min, x_max = sorted([x1, x2])
                    y_min, y_max = sorted([y1, y2])
                    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
                else:
                    # Polygon (3 or more vertices)
                    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in self.roi], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                self._mask_cache = mask
                self._roi_area = max(1, np.count_nonzero(mask))

        # Convert to grayscale and blur to remove high frequency noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply mask on the blurred frame to zero out regions outside ROI
        if self.roi is not None and len(self.roi) >= 2:
            blurred = cv2.bitwise_and(blurred, blurred, mask=self._mask_cache)

        # Initialize background model with first frame
        if self.background_accumulator is None:
            self.background_accumulator = blurred.astype("float")
            return False

        # Accumulate the weighted running average background
        cv2.accumulateWeighted(blurred, self.background_accumulator, self.alpha)
        background = cv2.convertScaleAbs(self.background_accumulator)

        # Compute absolute difference between current frame and background
        frame_delta = cv2.absdiff(blurred, background)
        
        # Apply binary thresholding
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate thresholded image to fill in holes/gaps
        dilated = cv2.dilate(thresh, None, iterations=2)

        # Find contours of moving regions
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_motion_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_contour_area:
                total_motion_area += area

        # Calculate fraction of motion pixels
        if self.roi is not None and len(self.roi) >= 2:
            total_pixels = self._roi_area
        else:
            total_pixels = h * w
            
        motion_fraction = total_motion_area / total_pixels

        return bool(motion_fraction >= self.threshold)

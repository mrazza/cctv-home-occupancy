import cv2
import numpy as np
from typing import Optional

class MotionDetector:
    def __init__(self, threshold: float = 0.005, min_contour_area: int = 500, alpha: float = 0.05):
        """
        Fast Motion Detector using frame differencing with running average background model.
        
        Args:
            threshold: Fraction of frame pixels that must be different to classify as motion (0.0 to 1.0)
            min_contour_area: Minimum area in pixels of a contour to count as moving object
            alpha: Accumulation rate for the running average background model (0.0 to 1.0)
        """
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.alpha = alpha
        self.background_accumulator: Optional[np.ndarray] = None

    def reset(self):
        """Resets the accumulated background model."""
        self.background_accumulator = None

    def detect(self, frame: np.ndarray) -> bool:
        """
        Processes a frame and returns True if motion is detected, updating the background.
        """
        # Convert to grayscale and blur to remove high frequency noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

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
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_fraction = total_motion_area / total_pixels

        return motion_fraction >= self.threshold

import cv2
import numpy as np
from typing import Tuple, Optional

def compute_luminance(frame: np.ndarray) -> float:
    """
    Computes the average luminance of a BGR image frame.
    Uses the grayscale intensity mean (standard ITU-R BT.601 luma).

    Args:
        frame: BGR image frame as a numpy array.

    Returns:
        Average luminance value between 0.0 (pitch black) and 255.0 (pure white).
    """
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def compute_blur_metric(frame: np.ndarray) -> float:
    """
    Computes the image sharpness/blur metric using the variance of the Laplacian.
    Higher values indicate sharp edges, lower values indicate blur or low texture.

    Args:
        frame: BGR image frame as a numpy array.

    Returns:
        Laplacian variance as a float.
    """
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8, denoise_prefilter: bool = True) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel of LAB color space.
    Enhances local contrast while avoiding amplification of high-frequency noise.

    Args:
        frame: BGR image frame.
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of the grid for histogram equalization (e.g. 8 for 8x8 tiles).
        denoise_prefilter: If True, applies lightweight 3x3 Gaussian pre-filtering on the L channel
            to prevent amplification of camera sensor shot noise on monochrome/IR night frames.

    Returns:
        Enhanced BGR image frame.
    """
    if frame is None or frame.size == 0:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    if denoise_prefilter:
        l_channel = cv2.GaussianBlur(l_channel, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size)))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_gamma(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Applies non-linear gamma correction via lookup table (LUT).
    Gamma > 1.0 brightens shadows and midtones while preserving dynamic range.

    Args:
        frame: BGR image frame.
        gamma: Gamma correction factor (> 0).

    Returns:
        Gamma-corrected BGR image frame.
    """
    if frame is None or frame.size == 0 or gamma <= 0.0:
        return frame
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(frame, table)

def enhance_low_light(
    frame: np.ndarray,
    method: str = 'clahe',
    clip_limit: float = 2.0,
    grid_size: int = 8,
    gamma: float = 1.5,
    denoise_prefilter: bool = True
) -> np.ndarray:
    """
    Enhances low-light image frame using the specified method.

    Args:
        frame: BGR image frame.
        method: 'clahe', 'gamma', 'both', or 'none'.
        clip_limit: CLAHE clip limit.
        grid_size: CLAHE grid size.
        gamma: Gamma correction factor.

    Returns:
        Enhanced BGR image frame.
    """
    if frame is None or frame.size == 0:
        return frame
    m = method.lower().strip()
    if m == 'clahe':
        return apply_clahe(frame, clip_limit=clip_limit, grid_size=grid_size, denoise_prefilter=denoise_prefilter)
    elif m == 'gamma':
        return apply_gamma(frame, gamma=gamma)
    elif m == 'both':
        g_frame = apply_gamma(frame, gamma=gamma)
        return apply_clahe(g_frame, clip_limit=clip_limit, grid_size=grid_size, denoise_prefilter=denoise_prefilter)
    return frame

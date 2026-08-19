import numpy as np
import cv2
from src.preprocessing import (
    compute_luminance,
    compute_blur_metric,
    apply_clahe,
    apply_gamma,
    enhance_low_light,
)

def test_compute_luminance():
    # Black image -> luminance 0
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert compute_luminance(black) == 0.0

    # White image -> luminance 255
    white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert compute_luminance(white) == 255.0

    # Gray image (128)
    gray = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert abs(compute_luminance(gray) - 128.0) < 1e-4

    # Empty frame
    assert compute_luminance(np.array([])) == 0.0
    assert compute_luminance(None) == 0.0

def test_compute_blur_metric():
    # Uniform image -> variance of Laplacian is 0
    uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert compute_blur_metric(uniform) == 0.0

    # Checkerboard image with high-frequency edges
    checker = np.zeros((100, 100, 3), dtype=np.uint8)
    checker[::2, ::2, :] = 255
    assert compute_blur_metric(checker) > 100.0

    # Empty frame
    assert compute_blur_metric(np.array([])) == 0.0
    assert compute_blur_metric(None) == 0.0

def test_apply_clahe():
    # Dark image with slight gradient
    dark = np.linspace(10, 40, 100, dtype=np.uint8)
    dark_frame = np.stack([dark]*100, axis=0)
    dark_bgr = cv2.cvtColor(dark_frame, cv2.COLOR_GRAY2BGR)

    enhanced = apply_clahe(dark_bgr, clip_limit=2.0, grid_size=8)
    assert enhanced.shape == dark_bgr.shape
    assert enhanced.dtype == np.uint8
    # CLAHE should boost local dynamic range / contrast
    assert np.mean(enhanced) > np.mean(dark_bgr) or np.std(enhanced) >= np.std(dark_bgr)

    # Empty frame
    assert apply_clahe(None) is None

def test_apply_gamma():
    # Mid-gray image (128)
    gray = np.ones((50, 50, 3), dtype=np.uint8) * 64
    brightened = apply_gamma(gray, gamma=2.0)
    assert np.mean(brightened) > np.mean(gray)

    darkened = apply_gamma(gray, gamma=0.5)
    assert np.mean(darkened) < np.mean(gray)

    # Empty frame / invalid gamma
    assert apply_gamma(None) is None
    assert np.array_equal(apply_gamma(gray, gamma=0.0), gray)

def test_enhance_low_light():
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 30

    # CLAHE
    c_out = enhance_low_light(frame, method="clahe")
    assert c_out.shape == frame.shape

    # Gamma
    g_out = enhance_low_light(frame, method="gamma", gamma=1.8)
    assert g_out.shape == frame.shape
    assert np.mean(g_out) > np.mean(frame)

    # Both
    b_out = enhance_low_light(frame, method="both", gamma=1.5)
    assert b_out.shape == frame.shape

    # None / unknown method
    n_out = enhance_low_light(frame, method="none")
    assert np.array_equal(n_out, frame)

    # Empty frame
    assert enhance_low_light(None) is None

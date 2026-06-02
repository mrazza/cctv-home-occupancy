import pytest
import numpy as np
from src.visualization import calculate_arrow_endpoint, compute_dead_zone_lines

def test_calculate_arrow_endpoint_normal():
    A = (0, 0)
    B = (10, 0)
    # Midpoint should be (5, 0)
    # Vector AB is (10, 0). Perpendicular vector pointing to the left side: (-0, 10) = (0, 10).
    # Normalized is (0.0, 1.0).
    # Arrow destination should be (5 + 50 * 0, 0 + 50 * 1) = (5, 50).
    mid, dest = calculate_arrow_endpoint(A, B, length=50.0)
    assert mid == (5, 0)
    assert dest == (5, 50)

def test_calculate_arrow_endpoint_zero_length():
    A = (10, 10)
    B = (10, 10)
    mid, dest = calculate_arrow_endpoint(A, B)
    assert mid == (10, 10)
    assert dest == (10, 10)

def test_compute_dead_zone_lines_normal():
    pt_A = (0, 0)
    pt_B = (10, 0)
    # Vector AB is (10, 0).
    # Unit normal pointing to the inside (+1 / left-hand side): (-dy, dx) / mag = (0, 10) / 10 = (0, 1)
    # Inside offset line: offset by +5 in y direction.
    # Outside offset line: offset by -5 in y direction.
    (ins_A, ins_B), (out_A, out_B) = compute_dead_zone_lines(pt_A, pt_B, dead_zone_half_px=5.0)
    assert ins_A == (0, 5)
    assert ins_B == (10, 5)
    assert out_A == (0, -5)
    assert out_B == (10, -5)

def test_compute_dead_zone_lines_zero_length():
    pt_A = (10, 10)
    pt_B = (10, 10)
    (ins_A, ins_B), (out_A, out_B) = compute_dead_zone_lines(pt_A, pt_B, dead_zone_half_px=5.0)
    assert ins_A == (10, 10)
    assert ins_B == (10, 10)
    assert out_A == (10, 10)
    assert out_B == (10, 10)

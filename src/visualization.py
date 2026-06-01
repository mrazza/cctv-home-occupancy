import numpy as np
from typing import Tuple


def calculate_arrow_endpoint(A: Tuple[int, int], B: Tuple[int, int], length: float = 50.0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given a line segment from A to B, calculates the center point of AB
    and an endpoint for an arrow pointing to the left-hand side of vector AB.
    """
    Ax, Ay = A
    Bx, By = B
    
    # Midpoint of AB
    Cx = int((Ax + Bx) / 2)
    Cy = int((Ay + By) / 2)
    
    # Vector AB
    vx = Bx - Ax
    vy = By - Ay
    
    # Perpendicular vector pointing to the left side: (-vy, vx)
    nx = -vy
    ny = vx
    
    # Normalize perpendicular vector
    mag = np.sqrt(nx**2 + ny**2)
    if mag < 1e-9:
        return (Cx, Cy), (Cx, Cy)
        
    ux = nx / mag
    uy = ny / mag
    
    # Arrow destination
    Dx = int(Cx + length * ux)
    Dy = int(Cy + length * uy)
    
    return (Cx, Cy), (Dx, Dy)


def compute_dead_zone_lines(
    pt_A: Tuple[int, int],
    pt_B: Tuple[int, int],
    dead_zone_half_px: float
) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Computes the two offset lines that form the dead zone boundaries.
    Returns ((inside_A, inside_B), (outside_A, outside_B)) in pixel coordinates.
    The inside line is offset toward the +1 (left/inside) side of directed vector AB.
    The outside line is offset toward the -1 (right/outside) side.
    """
    dx = pt_B[0] - pt_A[0]
    dy = pt_B[1] - pt_A[1]
    mag = np.sqrt(dx**2 + dy**2)
    if mag < 1e-9:
        return (pt_A, pt_B), (pt_A, pt_B)
    
    # Unit normal pointing to the inside (+1 / left-hand side): (-dy, dx) / mag
    nx = -dy / mag
    ny = dx / mag
    
    # Inside offset line (toward +1 side)
    inside_A = (int(pt_A[0] + nx * dead_zone_half_px), int(pt_A[1] + ny * dead_zone_half_px))
    inside_B = (int(pt_B[0] + nx * dead_zone_half_px), int(pt_B[1] + ny * dead_zone_half_px))
    
    # Outside offset line (toward -1 side)
    outside_A = (int(pt_A[0] - nx * dead_zone_half_px), int(pt_A[1] - ny * dead_zone_half_px))
    outside_B = (int(pt_B[0] - nx * dead_zone_half_px), int(pt_B[1] - ny * dead_zone_half_px))
    
    return (inside_A, inside_B), (outside_A, outside_B)

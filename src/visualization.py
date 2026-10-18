import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


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


def draw_zonal_tripwire(
    frame: np.ndarray,
    pt_A: Tuple[int, int],
    pt_B: Tuple[int, int],
    dead_zone_half_px: float = 0.0,
    show_labels: bool = True
) -> np.ndarray:
    """
    Draws the complete dual-zone state machine tripwire overlay:
    1. Primary centerline AB (thin cyan/blue).
    2. Threshold / Dead Zone polygon (semi-transparent amber fill).
    3. Boundary lines for Inside (+1) and Outside (-1) limits.
    4. Directional arrow pointing to the INSIDE / ENTER zone.
    5. Descriptive zone text labels reflecting the actual state machine boundaries.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w, _ = frame.shape
    
    # 1. Draw Dead Zone / Threshold polygon if width > 0
    if dead_zone_half_px > 0:
        (ins_A, ins_B), (out_A, out_B) = compute_dead_zone_lines(pt_A, pt_B, dead_zone_half_px)
        pts = np.array([ins_A, ins_B, out_B, out_A], dtype=np.int32)
        
        # Bounded crop overlay for fast alpha blending
        x_box, y_box, w_box, h_box = cv2.boundingRect(pts)
        x1 = max(0, x_box)
        y1 = max(0, y_box)
        x2 = min(w, x_box + w_box)
        y2 = min(h, y_box + h_box)
        
        if x2 > x1 and y2 > y1:
            sub = frame[y1:y2, x1:x2]
            overlay = sub.copy()
            local_pts = pts - np.array([x1, y1])
            cv2.fillPoly(overlay, [local_pts], (0, 215, 255)) # Amber/Gold
            cv2.addWeighted(overlay, 0.18, sub, 0.82, 0, sub)
            
        # Draw Inside boundary line (+1 boundary in light green)
        cv2.line(frame, ins_A, ins_B, (0, 230, 100), 1, cv2.LINE_AA)
        # Draw Outside boundary line (-1 boundary in orange/coral)
        cv2.line(frame, out_A, out_B, (0, 140, 255), 1, cv2.LINE_AA)

        if show_labels:
            # Place zone label inside the threshold zone
            mid_th = ((ins_A[0] + out_A[0]) // 2, (ins_A[1] + out_A[1]) // 2)
            cv2.putText(frame, "THRESHOLD (0)", (mid_th[0] + 5, mid_th[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    # 2. Draw Centerline AB
    cv2.line(frame, pt_A, pt_B, (255, 120, 0), 2, cv2.LINE_AA) # Blue-Cyan
    cv2.circle(frame, pt_A, 5, (0, 0, 255), -1) # Point A (Red)
    cv2.circle(frame, pt_B, 5, (255, 0, 0), -1) # Point B (Blue)
    
    cv2.putText(frame, "A", (pt_A[0] - 15, pt_A[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "B", (pt_B[0] + 8, pt_B[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # 3. Draw Perpendicular Direction Arrow pointing to INSIDE/ENTER (+1 side)
    mid, arrow_dest = calculate_arrow_endpoint(pt_A, pt_B, length=max(40.0, dead_zone_half_px + 25.0))
    cv2.arrowedLine(frame, mid, arrow_dest, (0, 255, 0), 3, tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.putText(frame, "INSIDE / ENTER (+1)", (arrow_dest[0] + 8, arrow_dest[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # 4. Optional Outside label pointing toward -1 side
    if show_labels:
        # Calculate opposite vector for Outside label
        ux = (arrow_dest[0] - mid[0])
        uy = (arrow_dest[1] - mid[1])
        out_label_pos = (mid[0] - int(ux * 0.7), mid[1] - int(uy * 0.7))
        cv2.putText(frame, "OUTSIDE (-1)", (out_label_pos[0] - 80, out_label_pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2, cv2.LINE_AA)

    return frame


def draw_tracked_person(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tracker_id: int,
    tracker_point: str = "bottom_center",
    zone_label: Optional[str] = None,
    conf: Optional[float] = None
) -> np.ndarray:
    """
    Draws a detected person bounding box along with their active tracking anchor point
    (bottom-center footprint contact point or centroid), track ID, confidence, and current zone.
    """
    if frame is None or frame.size == 0:
        return frame

    x1, y1, x2, y2 = bbox
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    
    # Calculate anchor point
    if tracker_point.lower().strip() == "bottom_center":
        ax = int((x1 + x2) / 2.0)
        ay = int(y2)
        anchor_name = "Foot"
    else:
        ax = int((x1 + x2) / 2.0)
        ay = int((y1 + y2) / 2.0)
        anchor_name = "Center"

    # Draw anchor marker: solid circle with crosshair
    cv2.circle(frame, (ax, ay), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (ax - 10, ay), (ax + 10, ay), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (ax, ay - 10), (ax, ay + 10), (0, 255, 255), 1, cv2.LINE_AA)

    # Label text
    label_parts = [f"ID {tracker_id}"]
    if conf is not None:
        label_parts.append(f"{conf:.2f}")
    if zone_label is not None:
        label_parts.append(f"[{zone_label}]")
    label_text = " | ".join(label_parts)

    (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tag_y = max(18, y1 - 6)
    cv2.rectangle(frame, (x1, tag_y - lh - 4), (x1 + lw + 6, tag_y + 2), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (x1 + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return frame

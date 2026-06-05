import os
import json
import cv2
import argparse
import numpy as np
from typing import Tuple, List, Optional
from src.config import CONFIG
from src.visualization import calculate_arrow_endpoint, compute_dead_zone_lines


def normalize_coordinates(A: Tuple[int, int], B: Tuple[int, int], width: int, height: int, sort_coords: bool = False) -> List[Tuple[float, float]]:
    """
    Converts raw pixel coordinates to normalized floats between 0.0 and 1.0.

    Args:
        A: Start point (x, y) in pixels.
        B: End point (x, y) in pixels.
        width: Frame width in pixels.
        height: Frame height in pixels.
        sort_coords: If True, sorts the coordinates so x1 <= x2 and y1 <= y2.

    Returns:
        List containing normalized [(x1, y1), (x2, y2)] coordinates.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive values")
        
    if sort_coords:
        x1, x2 = sorted([A[0], B[0]])
        y1, y2 = sorted([A[1], B[1]])
        A = (x1, y1)
        B = (x2, y2)
        
    return [
        (round(A[0] / width, 4), round(A[1] / height, 4)),
        (round(B[0] / width, 4), round(B[1] / height, 4))
    ]

def normalize_polygon(pts: List[Tuple[int, int]], width: int, height: int) -> List[Tuple[float, float]]:
    """
    Converts raw pixel coordinates of a polygon to normalized floats between 0.0 and 1.0.

    Args:
        pts: List of polygon vertices (x, y) in pixels.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of normalized [(x, y), ...] coordinates.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive values")
    return [(round(pt[0] / width, 4), round(pt[1] / height, 4)) for pt in pts]

def update_config_file(config_json_path: str, coordinates: List[Tuple[float, float]], key: str = "tripwire_line") -> bool:
    """
    Safely saves the coordinates to config.json under the specified key.

    Args:
        config_json_path: Path to the JSON configuration file to update.
        coordinates: Normalized coordinates to write.
        key: The key in the configuration JSON to set (e.g. "tripwire_line", "motion_roi").

    Returns:
        True if the write succeeded, False otherwise.
    """
    try:
        config_data = {}
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                try:
                    config_data = json.load(f)
                    if not isinstance(config_data, dict):
                        config_data = {}
                except Exception:
                    config_data = {}
                    
        config_data[key] = coordinates
        
        with open(config_json_path, "w") as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        print(f"[-] Error writing to config file: {e}")
        return False

class CalibrationApp:
    def __init__(self, stream_url: str, config_json_path: str = "config.json", mode: str = "tripwire"):
        """
        Initializes the CalibrationApp.

        Args:
            stream_url: Camera stream RTSP/video source URL.
            config_json_path: Path to JSON configuration file to modify.
            mode: "tripwire" or "roi" to determine what coordinates are being calibrated.
        """
        self.stream_url = stream_url
        self.config_json_path = config_json_path
        self.mode = mode
        self.cap = cv2.VideoCapture(stream_url)
        self.frame = None
        
        # State variables
        self.pt_A = None  # Tripwire Point A
        self.pt_B = None  # Tripwire Point B
        self.pts = []     # Polygon points
        self.polygon_closed = False
        self.mouse_pos = None
        self.is_drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        """
        OpenCV mouse event callback. Tracks clicks and mouse movement to record
        line segment points (tripwire) or vertices (ROI polygon).
        """
        if self.mode == "roi":
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.polygon_closed:
                    self.pts.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE:
                self.mouse_pos = (x, y)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.is_drawing and self.pt_A is None:
                    # First click: Point A
                    self.pt_A = (x, y)
                    self.is_drawing = True
                elif self.is_drawing:
                    # Second click: Point B
                    self.pt_B = (x, y)
                    self.is_drawing = False
                    
            elif event == cv2.EVENT_MOUSEMOVE:
                self.mouse_pos = (x, y)

    def draw_overlays(self, frame_draw) -> np.ndarray:
        """
        Draws interactive overlays, instruction HUD text, current lines/polygons,
        dead zones, and vector directions onto a copy of the camera frame.
        """
        h, w, _ = frame_draw.shape
        
        # Display instructions
        if self.mode == "roi":
            cv2.putText(frame_draw, "ROI POLYGON CALIBRATION MODE", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_draw, "Left click to add vertices sequentially.", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_draw, "Press [Enter] to Close Polygon | [S] to Save | [R] to Reset | [Q] to Quit", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_draw, "TRIPWIRE CALIBRATION MODE", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_draw, "Left click once for Point A (start), again for Point B (end)", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_draw, "Press [S] to Save | [R] to Reset | [Q] to Quit", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw status / elements
        if self.mode == "roi":
            num_pts = len(self.pts)
            if num_pts > 0:
                # Draw lines between consecutive points
                for i in range(num_pts - 1):
                    cv2.line(frame_draw, self.pts[i], self.pts[i+1], (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame_draw, self.pts[i], 5, (0, 0, 255), -1)
                cv2.circle(frame_draw, self.pts[-1], 5, (0, 0, 255), -1)
                
                if self.polygon_closed:
                    # Draw closing line
                    cv2.line(frame_draw, self.pts[-1], self.pts[0], (0, 255, 0), 2, cv2.LINE_AA)
                    # Semi-transparent overlay inside polygon
                    overlay = frame_draw.copy()
                    pts_arr = np.array(self.pts, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, frame_draw, 0.85, 0, frame_draw)
                    # Highlight green vertices
                    for p in self.pts:
                        cv2.circle(frame_draw, p, 5, (0, 255, 0), -1)
                    cv2.putText(frame_draw, "Status: Polygon Closed. Press [S] to Save or [R] to Reset", (15, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Draw preview line from last point to mouse
                    if self.mouse_pos:
                        cv2.line(frame_draw, self.pts[-1], self.mouse_pos, (0, 255, 255), 2, cv2.LINE_AA)
                        # Also draw preview closing line from mouse to first point
                        cv2.line(frame_draw, self.mouse_pos, self.pts[0], (0, 100, 255), 1, cv2.LINE_AA)
                    
                    status_text = f"Status: {num_pts} point(s) placed. Click next vertex, or press [Enter] to Close (requires >= 3 pts)"
                    cv2.putText(frame_draw, status_text, (15, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_draw, "Status: Left click to place first vertex", (15, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            if self.pt_A is None:
                cv2.putText(frame_draw, "Status: Click to place Point A", (15, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            elif self.is_drawing and self.mouse_pos:
                # Drawing preview
                cv2.line(frame_draw, self.pt_A, self.mouse_pos, (255, 100, 0), 2, cv2.LINE_AA)
                cv2.circle(frame_draw, self.pt_A, 5, (0, 0, 255), -1)
                cv2.putText(frame_draw, "Status: Click to place Point B", (15, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            elif self.pt_B:
                # Locked Tripwire
                cv2.line(frame_draw, self.pt_A, self.pt_B, (255, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(frame_draw, self.pt_A, 6, (0, 0, 255), -1)
                cv2.circle(frame_draw, self.pt_B, 6, (255, 0, 0), -1)
                
                # Label Point A & B
                cv2.putText(frame_draw, "A", (self.pt_A[0] - 15, self.pt_A[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_draw, "B", (self.pt_B[0] + 10, self.pt_B[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Calculate and draw left perpendicular arrow
                mid, arrow_dest = calculate_arrow_endpoint(self.pt_A, self.pt_B)
                cv2.arrowedLine(frame_draw, mid, arrow_dest, (0, 255, 0), 3, tipLength=0.3, line_type=cv2.LINE_AA)
                cv2.putText(frame_draw, "INSIDE / ENTER", (arrow_dest[0] + 10, arrow_dest[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw dead zone visualization
                dead_zone_width = CONFIG.tripwire_dead_zone_width
                if dead_zone_width > 0:
                    dead_zone_half_px = (dead_zone_width * h) / 2.0
                    (ins_A, ins_B), (out_A, out_B) = compute_dead_zone_lines(self.pt_A, self.pt_B, dead_zone_half_px)
                    
                    # Semi-transparent dead zone fill
                    overlay = frame_draw.copy()
                    zone_polygon = np.array([ins_A, ins_B, out_B, out_A], dtype=np.int32)
                    cv2.fillPoly(overlay, [zone_polygon], (0, 255, 255))  # Yellow fill
                    cv2.addWeighted(overlay, 0.15, frame_draw, 0.85, 0, frame_draw)
                    
                    # Thin boundary lines
                    cv2.line(frame_draw, ins_A, ins_B, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.line(frame_draw, out_A, out_B, (255, 255, 0), 1, cv2.LINE_AA)
                
                cv2.putText(frame_draw, "Status: Ready to Save [S] or Reset [R]", (15, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                            
        return frame_draw

    def run(self):
        """
        Starts the OpenCV UI window, listens for mouse clicks, and waits for user
        keyboard shortcuts (e.g. [S] to save, [R] to reset, [Q] to quit).
        """
        window_name = "ROI Calibration" if self.mode == "roi" else "Tripwire Calibration"
        print(f"[*] Opening stream for {self.mode} calibration...")
        if not self.cap.isOpened():
            print("[-] Error: Could not open camera stream.")
            return
            
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Aspect ratio correction parameters
        par_correction = 1.0
        if CONFIG.aspect_ratio_override is not None:
            # Computed dynamically on the first frame
            par_correction = None
        else:
            try:
                sar_num = self.cap.get(cv2.CAP_PROP_SAR_NUM)
                sar_den = self.cap.get(cv2.CAP_PROP_SAR_DEN)
                if isinstance(sar_num, (int, float)) and isinstance(sar_den, (int, float)) and sar_num > 0 and sar_den > 0:
                    par_correction = sar_num / sar_den
                    if abs(par_correction - 1.0) > 1e-4:
                        print(f"[*] Auto-detected non-square pixels from stream metadata. PAR correction factor: {par_correction:.4f}")
                    else:
                        par_correction = 1.0
                else:
                    par_correction = 1.0
            except Exception:
                par_correction = 1.0

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.frame is not None:
                    frame = self.frame.copy()
                else:
                    print("[-] Error: Failed to retrieve frame from stream.")
                    break
            else:
                # Apply aspect ratio correction
                if CONFIG.aspect_ratio_override is not None and par_correction is None:
                    h, w = frame.shape[:2]
                    par_correction = CONFIG.aspect_ratio_override / (w / h)
                    print(f"[*] Computed PAR correction from aspect ratio override: {par_correction:.4f} (Raw: {w}x{h}, Target: {CONFIG.aspect_ratio_override:.4f})")
                
                if par_correction is not None and abs(par_correction - 1.0) > 1e-4:
                    h, w = frame.shape[:2]
                    new_w = int(round(w * par_correction))
                    frame = cv2.resize(frame, (new_w, h), interpolation=cv2.INTER_LINEAR)

                self.frame = frame.copy()
                
            frame_draw = frame.copy()
            frame_draw = self.draw_overlays(frame_draw)
            
            cv2.imshow(window_name, frame_draw)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27: # Q or ESC
                print("[*] Calibration cancelled.")
                break
            elif key == ord('r'): # Reset
                self.pt_A = None
                self.pt_B = None
                self.pts = []
                self.polygon_closed = False
                self.is_drawing = False
                print(f"[*] {self.mode.capitalize()} coordinates reset.")
            elif key in (13, 10): # Enter
                if self.mode == "roi":
                    if len(self.pts) >= 3:
                        self.polygon_closed = True
                        print("[*] Polygon closed.")
                    else:
                        print("[-] Warning: Polygon requires at least 3 vertices to close!")
            elif key == ord('s'): # Save
                if self.mode == "roi":
                    if self.polygon_closed and len(self.pts) >= 3:
                        h, w, _ = frame.shape
                        norm_coords = normalize_polygon(self.pts, w, h)
                        print(f"[*] Normalized coordinates: {norm_coords}")
                        
                        success = update_config_file(self.config_json_path, norm_coords, key="motion_roi")
                        if success:
                            print(f"[+] Saved successfully to {self.config_json_path} under 'motion_roi'!")
                            break
                        else:
                            print(f"[-] Error: Could not update config file at {self.config_json_path}")
                    else:
                        print("[-] Warning: Polygon must be closed with [Enter] before saving!")
                else:
                    if self.pt_A and self.pt_B:
                        h, w, _ = frame.shape
                        norm_coords = normalize_coordinates(self.pt_A, self.pt_B, w, h)
                        print(f"[*] Normalized coordinates: {norm_coords}")
                        
                        success = update_config_file(self.config_json_path, norm_coords, key="tripwire_line")
                        if success:
                            print(f"[+] Saved successfully to {self.config_json_path} under 'tripwire_line'!")
                            break
                        else:
                            print(f"[-] Error: Could not update config file at {self.config_json_path}")
                    else:
                        print("[-] Warning: Complete drawing the tripwire line (Point A and B) before saving!")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Tripwire & ROI Calibration Tool")
    parser.add_argument("--rtsp", type=str, default=CONFIG.rtsp_url, help="RTSP stream URL")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON file")
    parser.add_argument("--mode", type=str, choices=["tripwire", "roi"], default="tripwire", help="Calibration mode: 'tripwire' or 'roi'")
    args = parser.parse_args()
    
    app = CalibrationApp(args.rtsp, config_json_path=args.config, mode=args.mode)
    app.run()

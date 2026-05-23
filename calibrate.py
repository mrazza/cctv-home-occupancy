import os
import json
import cv2
import argparse
import numpy as np
from typing import Tuple, List, Optional
from src.config import CONFIG

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

def normalize_coordinates(A: Tuple[int, int], B: Tuple[int, int], width: int, height: int) -> List[Tuple[float, float]]:
    """
    Converts raw pixel coordinates to normalized floats between 0.0 and 1.0.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive values")
        
    return [
        (round(A[0] / width, 4), round(A[1] / height, 4)),
        (round(B[0] / width, 4), round(B[1] / height, 4))
    ]

def update_config_file(config_json_path: str, line: List[Tuple[float, float]]) -> bool:
    """
    Safely saves the tripwire_line coordinates to config.json.
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
                    
        config_data["tripwire_line"] = line
        
        with open(config_json_path, "w") as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        print(f"[-] Error writing to config file: {e}")
        return False

class CalibrationApp:
    def __init__(self, stream_url: str, config_json_path: str = "config.json"):
        self.stream_url = stream_url
        self.config_json_path = config_json_path
        self.cap = cv2.VideoCapture(stream_url)
        self.frame = None
        
        # Line State
        self.pt_A = None
        self.pt_B = None
        self.mouse_pos = None
        self.is_drawing = False

    def mouse_callback(self, event, x, y, flags, param):
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
        h, w, _ = frame_draw.shape
        
        # Display instructions
        cv2.putText(frame_draw, "Left click once for Point A (start), again for Point B (end)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_draw, "Press [S] to Save | [R] to Reset | [Q] to Quit", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw status
        if self.pt_A is None:
            cv2.putText(frame_draw, "Status: Click to place Point A (Start)", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        elif self.is_drawing and self.mouse_pos:
            # Drawing preview line
            cv2.line(frame_draw, self.pt_A, self.mouse_pos, (255, 100, 0), 2, cv2.LINE_AA)
            cv2.circle(frame_draw, self.pt_A, 5, (0, 0, 255), -1)
            cv2.putText(frame_draw, "Status: Click to place Point B (End)", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        elif self.pt_B:
            # Locked Tripwire
            cv2.line(frame_draw, self.pt_A, self.pt_B, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(frame_draw, self.pt_A, 6, (0, 0, 255), -1)
            cv2.circle(frame_draw, self.pt_B, 6, (255, 0, -1), -1)
            
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
            
            cv2.putText(frame_draw, "Status: Ready to Save [S] or Reset [R]", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        
        return frame_draw

    def run(self):
        print("[*] Opening stream for calibration...")
        if not self.cap.isOpened():
            print("[-] Error: Could not open camera stream.")
            return
            
        cv2.namedWindow("Tripwire Calibration")
        cv2.setMouseCallback("Tripwire Calibration", self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.frame is not None:
                    frame = self.frame.copy()
                else:
                    print("[-] Error: Failed to retrieve frame from stream.")
                    break
            else:
                self.frame = frame.copy()
                
            frame_draw = frame.copy()
            frame_draw = self.draw_overlays(frame_draw)
            
            cv2.imshow("Tripwire Calibration", frame_draw)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27: # Q or ESC
                print("[*] Calibration cancelled.")
                break
            elif key == ord('r'): # Reset
                self.pt_A = None
                self.pt_B = None
                self.is_drawing = False
                print("[*] Tripwire coordinates reset.")
            elif key == ord('s'): # Save
                if self.pt_A and self.pt_B:
                    h, w, _ = frame.shape
                    norm_line = normalize_coordinates(self.pt_A, self.pt_B, w, h)
                    print(f"[*] Normalized line drawn: {norm_line}")
                    
                    success = update_config_file(self.config_json_path, norm_line)
                    if success:
                        print(f"[+] Saved successfully to {self.config_json_path}!")
                        break
                    else:
                        print(f"[-] Error: Could not update config file at {self.config_json_path}")
                else:
                    print("[-] Warning: Complete drawing the tripwire (Point A and B) before saving!")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Tripwire Calibration Tool")
    parser.add_argument("--rtsp", type=str, default=CONFIG.rtsp_url, help="RTSP stream URL")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON file")
    args = parser.parse_args()
    
    app = CalibrationApp(args.rtsp, config_json_path=args.config)
    app.run()

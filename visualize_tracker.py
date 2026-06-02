#!/usr/bin/env python3
import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
from typing import Tuple, List, Optional

# Load config and custom types/trackers
from src.config import CONFIG, load_config
from src.object_tracker import ObjectTracker
from src.visualization import calculate_arrow_endpoint, compute_dead_zone_lines
from src.pipeline import ThreadedVideoReader

# Define a premium color scheme (BGR format)
COLOR_CYAN = (255, 235, 0)
COLOR_PURPLE = (255, 0, 165)
COLOR_GOLD = (0, 235, 255)
COLOR_GREEN = (80, 230, 80)
COLOR_RED = (60, 60, 240)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (150, 150, 150)
COLOR_DARK_GRAY = (30, 30, 30)


def draw_hud_sidebar(frame, tracker, is_paused, show_roi, show_tripwire, show_history, fps, frame_count, source_name):
    h, w, _ = frame.shape
    
    # 1. Overlay panel (semi-transparent dark background)
    hud_w = 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (hud_w, h - 10), (15, 15, 15), -1)
    
    # Blend with original frame (alpha = 0.82)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    
    # Draw a thin stylish cyan border around the HUD panel
    cv2.rectangle(frame, (10, 10), (hud_w, h - 10), COLOR_CYAN, 1, cv2.LINE_AA)
    
    # 2. Text layout inside HUD
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 35
    
    def draw_text(text, color=COLOR_WHITE, scale=0.45, thickness=1):
        nonlocal y
        cv2.putText(frame, text, (25, y), font, scale, color, thickness, cv2.LINE_AA)
        y += 22

    def draw_header(text, color=COLOR_GOLD):
        nonlocal y
        y += 10
        cv2.putText(frame, text.upper(), (25, y), font, 0.55, color, 2, cv2.LINE_AA)
        y += 24
        # Draw a thin divider line
        cv2.line(frame, (25, y - 10), (hud_w - 20, y - 10), (60, 60, 60), 1)

    draw_header("CCTV OBJECT TRACKER HUD", color=COLOR_CYAN)
    
    # System Status
    draw_text(f"FPS: {fps:.1f}", color=COLOR_GREEN if fps > 5 else COLOR_RED, scale=0.45, thickness=2)
    draw_text(f"Frame Count: {frame_count}")
    
    state_str = "PAUSED" if is_paused else "PLAYING"
    state_col = COLOR_RED if is_paused else COLOR_GREEN
    draw_text(f"Playback State: {state_str}", color=state_col, scale=0.45, thickness=2)
    
    # Stream details
    trunc_src = source_name if len(source_name) < 28 else "..." + source_name[-25:]
    draw_text(f"Source: {trunc_src}", color=COLOR_GRAY)
    
    # Tracking Stats
    num_tracks = len(tracker.track_histories)
    draw_text(f"Active Tracks: {num_tracks}", color=COLOR_GOLD if num_tracks > 0 else COLOR_GRAY, scale=0.45, thickness=1 if num_tracks == 0 else 2)
    
    draw_header("TRACKING ENGINE")
    draw_text(f"Model Name: {os.path.basename(tracker.model.model_name)}")
    draw_text(f"YOLO Conf: {tracker.conf:.2f}")
    draw_text(f"Track Buffer: {tracker.track_buffer} frames")
    draw_text(f"Strict Segment: {tracker.tripwire_strict_segment}")
    
    draw_header("TRIPWIRE SETTINGS")
    dead_pct = int(tracker.dead_zone_width * 100)
    draw_text(f"Dead Zone: {tracker.dead_zone_width:.3f} ({dead_pct}%)")
    pt_a = tracker.tripwire_line[0]
    pt_b = tracker.tripwire_line[1]
    draw_text(f"Line Point A: ({pt_a[0]:.2f}, {pt_a[1]:.2f})")
    draw_text(f"Line Point B: ({pt_b[0]:.2f}, {pt_b[1]:.2f})")
    
    draw_header("VISUAL LAYERS")
    def get_toggle_str(val):
        return "ON" if val else "OFF"
    def get_toggle_col(val):
        return COLOR_GREEN if val else COLOR_GRAY
    
    draw_text(f"[M] Motion ROI:  {get_toggle_str(show_roi)}", color=get_toggle_col(show_roi))
    draw_text(f"[T] Tripwires:   {get_toggle_str(show_tripwire)}", color=get_toggle_col(show_tripwire))
    draw_text(f"[H] Trails:      {get_toggle_str(show_history)}", color=get_toggle_col(show_history))
    
    draw_header("SHORTCUTS")
    draw_text("[P] Pause / Resume Playback", color=COLOR_WHITE)
    draw_text("[R] Reset Tracker State", color=COLOR_WHITE)
    draw_text("[L] Live Reload config.json", color=COLOR_WHITE)
    draw_text("[Q / ESC] Close Utility", color=COLOR_WHITE)


def draw_history_trails(frame, tracker):
    for tid in list(tracker.track_histories.keys()):
        points = tracker.track_histories.get(tid, [])
        if len(points) < 2:
            continue
        
        # Render dynamic fading trail gradient
        num_pts = len(points)
        for i in range(num_pts - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            
            alpha = (i + 1) / num_pts
            thickness = max(1, int(alpha * 4))
            
            # Sleek purple gradient BGR blending
            b = int(240 + (255 - 240) * alpha)
            g = int(50 + (235 - 50) * alpha)
            r = int(140 + (0 - 140) * alpha)
            
            cv2.line(frame, pt1, pt2, (b, g, r), thickness, cv2.LINE_AA)


def draw_tracking_boxes(frame, tracker):
    if not hasattr(tracker, "latest_boxes") or tracker.latest_boxes is None:
        return
    
    boxes = tracker.latest_boxes
    if boxes.id is None:
        return
    
    xyxy = boxes.xyxy.cpu().numpy()
    tracker_ids = boxes.id.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    
    for i in range(len(tracker_ids)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        tid = tracker_ids[i]
        conf = confidences[i]
        
        # Bounding box tag style
        box_color = COLOR_CYAN
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        
        # Draw centroid
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, COLOR_RED, -1)
        
        # Overlay active labels
        label = f"ID: {tid} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Text header tag filled background
        header_y1 = max(0, y1 - label_h - 8)
        header_y2 = y1
        header_x1 = x1
        header_x2 = x1 + label_w + 10
        
        cv2.rectangle(frame, (header_x1, header_y1), (header_x2, header_y2), box_color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 4), font, font_scale, COLOR_DARK_GRAY, thickness, cv2.LINE_AA)


def draw_tripwires(frame, tracker):
    h, w, _ = frame.shape
    tx1, ty1 = int(tracker.tripwire_line[0][0] * w), int(tracker.tripwire_line[0][1] * h)
    tx2, ty2 = int(tracker.tripwire_line[1][0] * w), int(tracker.tripwire_line[1][1] * h)
    pt_A = (tx1, ty1)
    pt_B = (tx2, ty2)
    
    # Draw primary Tripwire Segment
    cv2.line(frame, pt_A, pt_B, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, pt_A, 6, (0, 0, 255), -1)  # Point A (Red)
    cv2.circle(frame, pt_B, 6, (255, 0, 0), -1)  # Point B (Blue)
    
    # Point Labels
    cv2.putText(frame, "A", (pt_A[0] - 15, pt_A[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "B", (pt_B[0] + 10, pt_B[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Draw Perpendicular Arrow pointing to INSIDE/ENTER (+1 side)
    mid, arrow_dest = calculate_arrow_endpoint(pt_A, pt_B)
    cv2.arrowedLine(frame, mid, arrow_dest, COLOR_GREEN, 3, tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.putText(frame, "INSIDE / ENTER", (arrow_dest[0] + 10, arrow_dest[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2, cv2.LINE_AA)
    
    # Draw Dead Zone bounding lines & transparent yellow fill
    dead_zone_width = tracker.dead_zone_width
    if dead_zone_width > 0:
        dead_zone_half_px = (dead_zone_width * h) / 2.0
        (ins_A, ins_B), (out_A, out_B) = compute_dead_zone_lines(pt_A, pt_B, dead_zone_half_px)
        
        overlay = frame.copy()
        zone_polygon = np.array([ins_A, ins_B, out_B, out_A], dtype=np.int32)
        cv2.fillPoly(overlay, [zone_polygon], COLOR_GOLD)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        cv2.line(frame, ins_A, ins_B, (0, 200, 220), 1, cv2.LINE_AA)
        cv2.line(frame, out_A, out_B, (0, 200, 220), 1, cv2.LINE_AA)


def draw_motion_roi(frame):
    if CONFIG.motion_roi is None:
        return
    
    h, w, _ = frame.shape
    pts = np.array([(int(pt[0] * w), int(pt[1] * h)) for pt in CONFIG.motion_roi], dtype=np.int32)
    
    # Subtle Sky Blue/Turquoise polygon
    roi_color = (255, 180, 100)
    cv2.polylines(frame, [pts], isClosed=True, color=roi_color, thickness=2, lineType=cv2.LINE_AA)
    
    if len(pts) > 0:
        cv2.putText(frame, "MOTION ROI", (pts[0][0] + 5, pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2, cv2.LINE_AA)


def reload_config_settings(config_path: str, tracker: ObjectTracker) -> bool:
    """Safely live-reloads configuration file changes and updates the active tracker parameters."""
    print(f"[*] Reloading configuration from {config_path}...")
    try:
        os.environ["CCTV_CONFIG_PATH"] = config_path
        
        # Load fresh configurations
        new_config = load_config()
        
        # Update existing CONFIG singleton
        for key, val in new_config.model_dump().items():
            setattr(CONFIG, key, val)
        
        # Apply updates to active ObjectTracker
        tracker.tripwire_line = CONFIG.tripwire_line
        tracker.dead_zone_width = CONFIG.tripwire_dead_zone_width
        tracker.conf = CONFIG.tracker_confidence
        tracker.tripwire_strict_segment = CONFIG.tripwire_strict_segment
        tracker.track_buffer = CONFIG.track_buffer
        
        # Re-create internal custom BoT-SORT YAML tracker config
        tracker._tracker_config_path = tracker._create_tracker_config(tracker.track_buffer, tracker.snapshot_dir)
        print("[+] Configuration reloaded successfully!")
        return True
    except Exception as e:
        print(f"[-] Error reloading configuration: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Tracking Visualization and Calibration Debugger")
    parser.add_argument("--rtsp", type=str, default=CONFIG.rtsp_url,
                        help="RTSP Stream URL or path to offline video file")
    parser.add_argument("--model", type=str, default=CONFIG.model_name,
                        help="YOLO model configuration name or local PT path")
    parser.add_argument("--conf", type=float, default=CONFIG.tracker_confidence,
                        help="Detection confidence threshold")
    parser.add_argument("--track-buffer", type=int, default=CONFIG.track_buffer,
                        help="Track buffer size (frame occlusion limit)")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to JSON configuration file")
    args = parser.parse_args()

    # Load initial config and set matching tracker settings
    print(f"[*] Initializing tracker utility with model: {args.model}")
    tracker = ObjectTracker(
        model_name=args.model,
        tripwire_line=CONFIG.tripwire_line,
        snapshot_dir=CONFIG.snapshot_dir,
        dead_zone_width=CONFIG.tripwire_dead_zone_width,
        tripwire_strict_segment=CONFIG.tripwire_strict_segment,
        conf=args.conf,
        track_buffer=args.track_buffer
    )

    is_live = args.rtsp.startswith("rtsp://") or args.rtsp.startswith("rtmp://")
    reader = None
    cap = None

    if is_live:
        print(f"[*] Connecting to live video stream: {args.rtsp}")
        reader = ThreadedVideoReader(args.rtsp).start()
    else:
        print(f"[*] Connecting to offline video file: {args.rtsp}")
        cap = cv2.VideoCapture(args.rtsp)
        if not cap.isOpened():
            print(f"[-] Error: Could not open video source '{args.rtsp}'")
            sys.exit(1)

    # State variables
    is_paused = False
    show_roi = True
    show_tripwire = True
    show_history = True
    
    frame_count = 0
    fps = 0.0
    last_fps_calc = time.time()
    fps_frames = 0
    
    window_name = "Antigravity YOLO Tracker Visualization Utility"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_frame = None

    while True:
        if not is_paused:
            if is_live:
                ret, frame = reader.read()
                if ret and frame is not None:
                    frame_count += 1
                    current_frame = frame.copy()
                    _ = tracker.process_frame(current_frame)
                    
                    fps_frames += 1
                    now = time.time()
                    if now - last_fps_calc >= 1.0:
                        fps = fps_frames / (now - last_fps_calc)
                        fps_frames = 0
                        last_fps_calc = now
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[*] End of stream or failed to grab frame.")
                    print("[*] Rewinding offline video playback...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
                current_frame = frame.copy()
                _ = tracker.process_frame(current_frame)
                
                fps_frames += 1
                now = time.time()
                if now - last_fps_calc >= 1.0:
                    fps = fps_frames / (now - last_fps_calc)
                    fps_frames = 0
                    last_fps_calc = now

        if current_frame is None:
            # Wait a bit for the first frame
            time.sleep(0.01)
            continue

        # Draw overlays on top of current frame copy
        frame_draw = current_frame.copy()
        
        # 1. Render active tracks and bounding boxes
        draw_tracking_boxes(frame_draw, tracker)
        
        # 2. Render track history trails
        if show_history:
            draw_history_trails(frame_draw, tracker)
            
        # 3. Render Motion ROI polygon
        if show_roi:
            draw_motion_roi(frame_draw)
            
        # 4. Render tripwires and dead zones
        if show_tripwire:
            draw_tripwires(frame_draw, tracker)
            
        # 5. Render HUD sidebar on left
        draw_hud_sidebar(
            frame_draw, tracker, is_paused, show_roi, show_tripwire, show_history,
            fps, frame_count, args.rtsp
        )

        cv2.imshow(window_name, frame_draw)
        
        # Keyboard handling (Wait 30ms or adjust based on performance)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            print("[*] Exiting visualization utility...")
            break
            
        elif key == ord('p'):  # P - Pause/Resume Toggle
            is_paused = not is_paused
            print(f"[*] Playback {'paused' if is_paused else 'resumed'}")
            
        elif key == ord('r'):  # R - Reset Tracker State
            tracker.track_histories.clear()
            tracker.track_confirmed_sides.clear()
            tracker.track_last_seen.clear()
            print("[*] Object tracker state and histories cleared.")
            
        elif key == ord('m'):  # M - Toggle Motion ROI Rendering
            show_roi = not show_roi
            print(f"[*] Motion ROI layer visibility: {show_roi}")
            
        elif key == ord('t'):  # T - Toggle Tripwires Layer Rendering
            show_tripwire = not show_tripwire
            print(f"[*] Tripwire visual layer visibility: {show_tripwire}")
            
        elif key == ord('h'):  # H - Toggle Track History Trails Rendering
            show_history = not show_history
            print(f"[*] Tracking history trails layer visibility: {show_history}")
            
        elif key == ord('l'):  # L - Live Reload Configuration
            success = reload_config_settings(args.config, tracker)
            if not success:
                print("[-] Failed to live reload settings from configuration file.")

    if is_live:
        reader.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

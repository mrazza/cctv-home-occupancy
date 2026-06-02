#!/usr/bin/env python3
import os
import sys
import logging
import cv2
import time
import argparse
import numpy as np

logger = logging.getLogger("visualize_tracker")

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
    """
    Draws a stylish, semi-transparent HUD sidebar panel on the left side of the frame.
    Displays real-time performance statistics, active tracker parameters, tripwire settings,
    visual layer toggle indicators, and list of keyboard shortcuts.
    """
    h, w, _ = frame.shape
    hud_w = max(240, min(450, int(w * 0.18)))
    
    # 1. Overlay panel (semi-transparent dark background) - Optimized ROI crop/blend
    sub_frame = frame[10:h - 10, 10:hud_w]
    overlay = sub_frame.copy()
    cv2.rectangle(overlay, (0, 0), (hud_w - 10, h - 20), (15, 15, 15), -1)
    
    # Blend with original sub-frame view
    cv2.addWeighted(overlay, 0.82, sub_frame, 0.18, 0, sub_frame)
    
    # Draw a thin stylish cyan border around the HUD panel
    cv2.rectangle(frame, (10, 10), (hud_w, h - 10), COLOR_CYAN, 1, cv2.LINE_AA)
    
    # Calculate scale factors based on the frame height
    scale_factor = h / 1080.0
    is_compact = h < 700
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if is_compact:
        font_scale = max(0.32, min(0.42, 0.38 * (h / 500.0)))
        header_scale = font_scale + 0.08
        line_height = int(32 * font_scale)
        y = int(35 * scale_factor)
        if y < 15:
            y = 15
        x_offset = int(25 * (hud_w / 340.0))
        if x_offset < 12:
            x_offset = 12
    else:
        font_scale = max(0.45, min(0.65, 0.45 * scale_factor * 1.1))
        header_scale = font_scale + 0.1
        line_height = int(45 * font_scale)
        y = int(35 * scale_factor)
        if y < 35:
            y = 35
        x_offset = int(25 * (hud_w / 340.0))
        if x_offset < 20:
            x_offset = 20
            
    def draw_text(text, color=COLOR_WHITE, scale=font_scale, thickness=None):
        nonlocal y
        if thickness is None:
            thickness = 2 if scale >= 0.45 and (color in (COLOR_GREEN, COLOR_RED, COLOR_GOLD)) else 1
        cv2.putText(frame, text, (x_offset, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_height

    def draw_header(text, color=COLOR_GOLD):
        nonlocal y
        y += int(10 * scale_factor) if not is_compact else 4
        header_thick = 2 if header_scale >= 0.45 else 1
        cv2.putText(frame, text.upper(), (x_offset, y), font, header_scale, color, header_thick, cv2.LINE_AA)
        y += line_height + (4 if not is_compact else 2)
        # Draw a thin divider line
        divider_y = y - (int(10 * scale_factor) if not is_compact else 6)
        cv2.line(frame, (x_offset, divider_y), (hud_w - 20, divider_y), (60, 60, 60), 1)

    def get_toggle_str(val):
        return "ON" if val else "OFF"
    def get_toggle_col(val):
        return COLOR_GREEN if val else COLOR_GRAY

    if is_compact:
        draw_header("CCTV TRACKER HUD", color=COLOR_CYAN)
        
        # System Status
        state_str = "PAUSED" if is_paused else "PLAYING"
        state_col = COLOR_RED if is_paused else COLOR_GREEN
        draw_text(f"FPS: {fps:.1f} | Frame: {frame_count}")
        draw_text(f"State: {state_str}", color=state_col)
        
        # Tracking Stats & Details
        num_tracks = len(tracker.track_histories)
        draw_text(f"Tracks: {num_tracks}", color=COLOR_GOLD if num_tracks > 0 else COLOR_GRAY)
        
        model_base = os.path.basename(tracker.model.model_name)
        draw_text(f"Model: {model_base}")
        draw_text(f"YOLO Conf: {tracker.conf:.2f} | Buf: {tracker.track_buffer}")
        
        # Tripwire
        dead_pct = int(tracker.dead_zone_width * 100)
        pt_a = tracker.tripwire_line[0]
        pt_b = tracker.tripwire_line[1]
        draw_text(f"Dead Zone: {dead_pct}%")
        draw_text(f"A:({pt_a[0]:.2f},{pt_a[1]:.2f}) B:({pt_b[0]:.2f},{pt_b[1]:.2f})", color=COLOR_GRAY)
        
        # Visual Layers
        draw_text(f"ROI:{get_toggle_str(show_roi)} | Trip:{get_toggle_str(show_tripwire)} | Trails:{get_toggle_str(show_history)}")
        
        # Shortcuts
        draw_text("P:Pause | R:Reset", color=COLOR_WHITE)
        draw_text("L:Reload | Q:Quit", color=COLOR_WHITE)
    else:
        # Full Layout
        draw_header("CCTV OBJECT TRACKER HUD", color=COLOR_CYAN)
        
        # System Status
        draw_text(f"FPS: {fps:.1f}", color=COLOR_GREEN if fps > 5 else COLOR_RED, scale=font_scale, thickness=2 if font_scale >= 0.45 else 1)
        draw_text(f"Frame Count: {frame_count}")
        
        state_str = "PAUSED" if is_paused else "PLAYING"
        state_col = COLOR_RED if is_paused else COLOR_GREEN
        draw_text(f"Playback State: {state_str}", color=state_col, scale=font_scale, thickness=2 if font_scale >= 0.45 else 1)
        
        # Stream details
        trunc_src = source_name if len(source_name) < 28 else "..." + source_name[-25:]
        draw_text(f"Source: {trunc_src}", color=COLOR_GRAY)
        
        # Tracking Stats
        num_tracks = len(tracker.track_histories)
        draw_text(f"Active Tracks: {num_tracks}", color=COLOR_GOLD if num_tracks > 0 else COLOR_GRAY, scale=font_scale, thickness=1 if num_tracks == 0 else 2)
        
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
        draw_text(f"[M] Motion ROI:  {get_toggle_str(show_roi)}", color=get_toggle_col(show_roi))
        draw_text(f"[T] Tripwires:   {get_toggle_str(show_tripwire)}", color=get_toggle_col(show_tripwire))
        draw_text(f"[H] Trails:      {get_toggle_str(show_history)}", color=get_toggle_col(show_history))
        
        draw_header("SHORTCUTS")
        draw_text("[P] Pause / Resume Playback", color=COLOR_WHITE)
        draw_text("[R] Reset Tracker State", color=COLOR_WHITE)
        draw_text("[L] Live Reload config.json", color=COLOR_WHITE)
        draw_text("[Q / ESC] Close Utility", color=COLOR_WHITE)


def draw_history_trails(frame, tracker):
    """
    Draws fading centroid history trails for all currently tracked objects.
    Uses BGR color blending to create a sleek purple-to-pink gradient effect.
    """
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
    """
    Draws bounding boxes, centroids, track IDs, and confidence scores for active detections.
    Also draws a filled background tag header above the bounding boxes.
    """
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
    """
    Draws the primary tripwire segment, its directional inside entry arrow,
    and a semi-transparent yellow dead zone area bounding box.
    """
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
        
        # Optimized: crop polygon bounding box ROI to avoid full-frame copy/blend
        pts = np.array([ins_A, ins_B, out_B, out_A], dtype=np.int32)
        x_box, y_box, w_box, h_box = cv2.boundingRect(pts)
        
        x1 = max(0, x_box)
        y1 = max(0, y_box)
        x2 = min(w, x_box + w_box)
        y2 = min(h, y_box + h_box)
        
        if x2 > x1 and y2 > y1:
            sub_frame = frame[y1:y2, x1:x2]
            overlay = sub_frame.copy()
            local_pts = pts - np.array([x1, y1])
            cv2.fillPoly(overlay, [local_pts], COLOR_GOLD)
            cv2.addWeighted(overlay, 0.15, sub_frame, 0.85, 0, sub_frame)
        
        cv2.line(frame, ins_A, ins_B, (0, 200, 220), 1, cv2.LINE_AA)
        cv2.line(frame, out_A, out_B, (0, 200, 220), 1, cv2.LINE_AA)


def draw_motion_roi(frame):
    """
    Draws the motion detection Region of Interest (ROI) boundary polygon.
    """
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
    logger.info(f"Reloading configuration from {config_path}...")
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
        tracker.yolo_imgsz = CONFIG.yolo_imgsz
        tracker.yolo_device = CONFIG.yolo_device
        
        # Re-create internal custom BoT-SORT YAML tracker config
        tracker._tracker_config_path = tracker._create_tracker_config(tracker.track_buffer, tracker.snapshot_dir)
        logger.info("Configuration reloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error reloading configuration: {e}")
        return False


def main():
    """
    CLI Entry point. Initializes ObjectTracker, connects to RTSP stream or offline file,
    runs the YOLO processing, renders HUD and visualization overlays, and processes keyboard shortcuts.
    """
    from src.logger import setup_logging
    setup_logging(log_level="INFO")

    parser = argparse.ArgumentParser(description="YOLO Object Tracking Visualization and Calibration Debugger")
    parser.add_argument("--rtsp", type=str, default=CONFIG.rtsp_url,
                        help="RTSP Stream URL or path to offline video file")
    parser.add_argument("--model", type=str, default=CONFIG.model_name,
                        help="YOLO model configuration name or local PT path")
    parser.add_argument("--conf", type=float, default=CONFIG.tracker_confidence,
                        help="Detection confidence threshold")
    parser.add_argument("--track-buffer", type=int, default=CONFIG.track_buffer,
                        help="Track buffer size (frame occlusion limit)")
    parser.add_argument("--buffer-size", type=int, default=CONFIG.video_buffer_size,
                        help="OpenCV VideoCapture buffer size")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to JSON configuration file")
    parser.add_argument("--yolo-imgsz", type=int, default=CONFIG.yolo_imgsz,
                        help="YOLO inference image size (default from config)")
    parser.add_argument("--yolo-device", type=str, default=CONFIG.yolo_device,
                        help="Device to run YOLO on (e.g. 'cpu', 'cuda', '0')")
    args = parser.parse_args()

    # Load initial config and set matching tracker settings
    logger.info(f"Initializing tracker utility with model: {args.model}")
    tracker = ObjectTracker(
        model_name=args.model,
        tripwire_line=CONFIG.tripwire_line,
        snapshot_dir=CONFIG.snapshot_dir,
        dead_zone_width=CONFIG.tripwire_dead_zone_width,
        tripwire_strict_segment=CONFIG.tripwire_strict_segment,
        conf=args.conf,
        track_buffer=args.track_buffer,
        yolo_imgsz=args.yolo_imgsz,
        yolo_device=args.yolo_device
    )

    is_live = args.rtsp.startswith("rtsp://") or args.rtsp.startswith("rtmp://")
    reader = None
    cap = None
    video_fps = 30.0

    if is_live:
        logger.info(f"Connecting to live video stream: {args.rtsp} with buffer size: {args.buffer_size}")
        reader = ThreadedVideoReader(args.rtsp, buffer_size=args.buffer_size).start()
    else:
        logger.info(f"Connecting to offline video file: {args.rtsp}")
        cap = cv2.VideoCapture(args.rtsp)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video source '{args.rtsp}'")
            sys.exit(1)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps) or video_fps > 120.0:
            video_fps = 30.0

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
        loop_start_time = time.time()
        has_new_frame = False
        
        if not is_paused:
            if is_live:
                t_read_start = time.time()
                ret, frame = reader.read()
                t_read = time.time() - t_read_start
                if t_read > 0.1:
                    logger.warning(f"reader.read() took {t_read * 1000.0:.1f}ms")

                if ret and frame is not None:
                    has_new_frame = True
                    frame_count += 1
                    current_frame = frame.copy()
                    
                    t_track_start = time.time()
                    _ = tracker.process_frame(current_frame)
                    t_track = time.time() - t_track_start
                    if t_track > 0.1:
                        logger.warning(f"tracker.process_frame() took {t_track * 1000.0:.1f}ms")
                    
                    fps_frames += 1
                    now = time.time()
                    if now - last_fps_calc >= 1.0:
                        fps = fps_frames / (now - last_fps_calc)
                        fps_frames = 0
                        last_fps_calc = now
            else:
                t_read_start = time.time()
                ret, frame = cap.read()
                t_read = time.time() - t_read_start
                if t_read > 0.1:
                    logger.warning(f"cap.read() took {t_read * 1000.0:.1f}ms")

                if not ret or frame is None:
                    logger.info("End of stream or failed to grab frame.")
                    logger.info("Rewinding offline video playback...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                has_new_frame = True
                frame_count += 1
                current_frame = frame.copy()
                
                t_track_start = time.time()
                _ = tracker.process_frame(current_frame)
                t_track = time.time() - t_track_start
                if t_track > 0.1:
                    logger.warning(f"tracker.process_frame() took {t_track * 1000.0:.1f}ms")
                
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

        # If it is live mode, we are not paused, and no new frame arrived:
        # Pump GUI events briefly and sleep to avoid CPU-starving busy loops
        if is_live and not is_paused and not has_new_frame:
            t_wait_start = time.time()
            key = cv2.waitKey(10) & 0xFF
            t_wait = time.time() - t_wait_start
            if t_wait > 0.1:
                logger.warning(f"cv2.waitKey(10) took {t_wait * 1000.0:.1f}ms")
        else:
            # Draw overlays on top of current frame copy
            frame_draw = current_frame.copy()
            
            t_render_start = time.time()
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
            t_render = time.time() - t_render_start
            if t_render > 0.1:
                logger.warning(f"rendering/imshow took {t_render * 1000.0:.1f}ms")
            
            # Determine appropriate wait delay
            if is_live:
                wait_time = 1
            else:
                elapsed_ms = (time.time() - loop_start_time) * 1000.0
                wait_time = max(1, int((1000.0 / video_fps) - elapsed_ms))
                
            t_wait_start = time.time()
            key = cv2.waitKey(wait_time) & 0xFF
            t_wait = time.time() - t_wait_start
            if t_wait > 0.1:
                logger.warning(f"cv2.waitKey({wait_time}) took {t_wait * 1000.0:.1f}ms")
        
        if key == ord('q') or key == 27:  # Q or ESC
            logger.info("Exiting visualization utility...")
            break
            
        elif key == ord('p'):  # P - Pause/Resume Toggle
            is_paused = not is_paused
            logger.info(f"Playback {'paused' if is_paused else 'resumed'}")
            
        elif key == ord('r'):  # R - Reset Tracker State
            tracker.track_histories.clear()
            tracker.track_confirmed_sides.clear()
            tracker.track_last_seen.clear()
            logger.info("Object tracker state and histories cleared.")
            
        elif key == ord('m'):  # M - Toggle Motion ROI Rendering
            show_roi = not show_roi
            logger.info(f"Motion ROI layer visibility: {show_roi}")
            
        elif key == ord('t'):  # T - Toggle Tripwires Layer Rendering
            show_tripwire = not show_tripwire
            logger.info(f"Tripwire visual layer visibility: {show_tripwire}")
            
        elif key == ord('h'):  # H - Toggle Track History Trails Rendering
            show_history = not show_history
            logger.info(f"Tracking history trails layer visibility: {show_history}")
            
        elif key == ord('l'):  # L - Live Reload Configuration
            success = reload_config_settings(args.config, tracker)
            if not success:
                logger.error("Failed to live reload settings from configuration file.")

    if is_live:
        reader.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

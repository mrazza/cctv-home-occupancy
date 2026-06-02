# House Presence Monitoring System (cctv-home-occupancy)

A private, high-performance local Linux pipeline to monitor presence in your home using realtime video streams (e.g. Nest cameras). It operates in a **"Fast & Slow"** fashion to optimize CPU utilization:
1. **Fast Stage (0% CPU idle)**: Continuous, ultra-lightweight pixel differencing with OpenCV.
2. **Slow Stage (Active AI)**: Activates YOLOv8 and ByteTrack to track people crossing virtual tripwires only when motion is detected.

Exposes a FastAPI endpoint that can be directly queried or hooked into a Mattermost Chatbot Slash Command.

---

## 🛠 Features
- **RTSP Input**: Integrated with local WebRTC-to-RTSP bridges (e.g. Scrypted or go2rtc).
- **CPU Optimized**: Uses background subtraction to wake/sleep the deep learning tracking logic.
- **Directional Crossing Math**: Uses geometric intersection and vector cross-product to detect entering vs. leaving.
- **Self-Correcting State Engine**: Maintains transactional SQLite DB representing household occupancy.
- **Queryable API**: FastAPI allows Mattermost bots to query "Is anyone home?" or fetch snapshots of recent entry/exit events.
- **Facial Crop Support**: Automatically isolates crops of faces/bodies for future facial identification.
- **Configurable Event Webhooks**: Instantly dispatches POST requests to an array of URLs when a tripwire crossing event is detected.

---

## 📐 Configuring Your Tripwire (Line-Crossing)

Every camera has a unique angle, so you must define the doorway boundary in `src/config.py` using normalized coordinates from `0.0` (top/left) to `1.0` (bottom/right).

### The "Left-Hand Rule" for Directed Vectors
The tripwire is represented as a directed vector starting at Point A $(x_1, y_1)$ and ending at Point B $(x_2, y_2)$. 

To determine which side of the line is **Inside** (incrementing occupancy) vs. **Outside** (decrementing occupancy), use the **Left-Hand Rule**:
> **Imagine standing at Point A and looking down the line towards Point B:**
> * Any crossing onto your **Left-hand side** is classified as **Inside (+1 / ENTER)**.
> * Any crossing onto your **Right-hand side** is classified as **Outside (-1 / LEAVE)**.

---

### Concrete Visual Examples (Screen Space)
In screen coordinates, the top-left corner is `(0.0, 0.0)` and the bottom-right corner is `(1.0, 1.0)`. Here is how the line direction maps visually:

#### 1. Horizontal Doorways
* **Drawing from Left to Right:** Point A is Left, Point B is Right.
  * Stand at A, look at B: Your Left hand points **downward**.
  * 🟢 **Inside (+1 / ENTER)** is visually **below** the line (higher Y values, closer to the bottom of the screen).
  * 🔴 **Outside (-1 / LEAVE)** is visually **above** the line (lower Y values, closer to the top of the screen).
* **Drawing from Right to Left:** Point A is Right, Point B is Left.
  * Stand at A, look at B: Your Left hand points **upward**.
  * 🟢 **Inside (+1 / ENTER)** is visually **above** the line (closer to the top of the screen).
  * 🔴 **Outside (-1 / LEAVE)** is visually **below** the line (closer to the bottom of the screen).

#### 2. Vertical Doorways
* **Drawing from Bottom to Top:** Point A is Bottom, Point B is Top.
  * Stand at A, look at B: Your Left hand points to the **left**.
  * 🟢 **Inside (+1 / ENTER)** is visually to the **Left** of the line (lower X values).
  * 🔴 **Outside (-1 / LEAVE)** is visually to the **Right** of the line (higher X values).
* **Drawing from Top to Bottom:** Point A is Top, Point B is Bottom.
  * Stand at A, look at B: Your Left hand points to the **right**.
  * 🟢 **Inside (+1 / ENTER)** is visually to the **Right** of the line (higher X values).
  * 🔴 **Outside (-1 / LEAVE)** is visually to the **Left** of the line (lower X values).

---

### 3. Hysteresis Dead Zone
To prevent centroid jitter from causing spurious, rapid-fire `ENTER` and `LEAVE` events (e.g. when someone stands on or lingers near the tripwire), the tracker employs a configurable **hysteresis dead zone** around the tripwire.
- The dead zone is defined as a fraction of the frame height (via `tripwire_dead_zone_width`, default `0.05` or 5%).
- A track's side state is only updated/committed (and an event emitted) when the person's centroid **fully clears** the dead zone boundary on the opposite side of the line.
- Movement inside the dead zone is ignored and does not alter the confirmed side state.
- Set the dead zone width to `0.0` to disable this behavior and trigger events immediately upon any crossing.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Linux Server with Python 3.10+ and Docker.
- A local RTSP stream from your Nest Camera using a WebRTC-to-RTSP bridge (e.g., **Scrypted** or **go2rtc**).

### 2. Install Dependencies
```bash
cd sources/mrazza/cctv-home-occupancy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Unit and Integration Tests
To make sure all tracking math, state engines, and API endpoints function perfectly:
```bash
PYTHONPATH=src pytest
```

---

## ⚙️ Configuration

The system is highly configurable. You can specify settings via a JSON configuration file (e.g., `config.json`) and/or individual environment variables.

### Precedence Order
1. **Environment Variables**: Individual environment variables (e.g., `CCTV_RTSP_URL`) override JSON configuration and default values.
2. **JSON Config File**: Values loaded from `config.json` (or the path defined by the `CCTV_CONFIG_PATH` environment variable) override defaults.
3. **Defaults**: Standard fallback configurations defined in [src/config.py](src/config.py).

---

### Configuration Options Reference

#### 🎥 Video & Stream Settings

| JSON Key | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `rtsp_url` | `CCTV_RTSP_URL` | `rtsp://localhost:8554/nest-cam` | RTSP URL for the camera stream. |
| `fps_limit` | `CCTV_FPS_LIMIT` | `10` | Target frames per second to process. |
| `video_buffer_size` | `CCTV_VIDEO_BUFFER_SIZE` | `1` | Size of the OpenCV `VideoCapture` buffer queue. |

#### 🧠 YOLO & Tracking Settings

| JSON Key | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `CCTV_MODEL_NAME` | `yolov8n.pt` | YOLO model configuration name or local `.pt` path. |
| `yolo_imgsz` | `CCTV_YOLO_IMGSZ` | `640` | Image size (resolution) for YOLO inference. |
| `yolo_device` | `CCTV_YOLO_DEVICE` | `null` | Device to run YOLO model on (e.g., `'cpu'`, `'cuda'`, `'0'`). |
| `tracker_confidence` | `CCTV_TRACKER_CONFIDENCE` | `0.1` | Minimum detection confidence threshold for YOLO person detections. |
| `track_buffer` | `CCTV_TRACK_BUFFER` | `30` | Number of frames to keep lost tracks alive before reassigning a new tracker ID. At 10 FPS, defaults to 3 seconds. |

#### 🏃 Motion Detection Settings (Fast Stage)

| JSON Key | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `motion_threshold` | `CCTV_MOTION_THRESHOLD` | `0.005` | Fraction of changed pixels (0.0 to 1.0) to trigger motion state. |
| `motion_min_contour_area` | `CCTV_MIN_CONTOUR_AREA` | `500` | Minimum pixel area of a moving contour to trigger motion state. |
| `background_alpha` | `CCTV_BACKGROUND_ALPHA` | `0.05` | Accumulator update speed (alpha) for background subtraction. |
| `motion_cooldown_frames` | `CCTV_MOTION_COOLDOWN` | `150` | Number of idle frames before returning to the fast motion detection stage. |
| `motion_roi` | `CCTV_MOTION_ROI` | `null` | Region of interest for motion detection. Array of normalized coordinates, e.g. `[[x1,y1],[x2,y2],...]`. If `null`, uses full frame. |

#### 📐 Tripwire & Line-Crossing Settings

| JSON Key | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `tripwire_line` | `CCTV_TRIPWIRE_LINE` | `[[0.2, 0.5], [0.8, 0.5]]` | Coordinates of the tripwire line segment `[[x1, y1], [x2, y2]]` normalized between `0.0` and `1.0`. |
| `tripwire_dead_zone_width` | `CCTV_DEAD_ZONE_WIDTH` | `0.05` | Width of the hysteresis dead zone around the tripwire, as a fraction of frame height. |
| `tripwire_strict_segment` | `CCTV_TRIPWIRE_STRICT_SEGMENT` | `false` | If `true`, requires the centroid projection to fall strictly within the tripwire line segment to register a crossing. |

#### 🌐 API Server & Stream Control Settings

| JSON Key | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `host` | `CCTV_HOST` | `0.0.0.0` | IP address to bind the API server. |
| `port` | `CCTV_PORT` | `8000` | Port to bind the API server. |
| `trigger_mode` | `CCTV_TRIGGER_MODE` | `continuous` | Trigger mode: `'continuous'` (processing active) or `'event'` (keeps YOLO/ByteTrack stream active after trigger event). |
| `event_stream_duration` | `CCTV_EVENT_STREAM_DURATION` | `45` | Duration in seconds to keep stream active after a trigger event in `'event'` trigger mode. |

#### 📝 Logging, Database, & Storage Settings

| JSON Key | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `log_level` | `CCTV_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). |
| `log_file` | `CCTV_LOG_FILE` | `logs/cctv.log` | Path to log file. Use `null` or empty to disable file logging. |
| `db_path` | `CCTV_DB_PATH` | `db/presence.db` | Path to SQLite presence database file. |
| `snapshot_dir` | `CCTV_SNAPSHOT_DIR` | `snapshots` | Directory path where face/body crops are stored. |

#### 🔗 Webhook Settings

| JSON Key | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `webhook_urls` | `CCTV_WEBHOOK_URLS` | `[]` | List/Array of webhook URLs (or comma-separated list/JSON array via Env) to trigger on events. |
| `webhook_timeout` | `CCTV_WEBHOOK_TIMEOUT` | `5` | Timeout in seconds for webhook requests. |

---

## 🏃 Running the Application

To start both the API server and the background camera-monitoring pipeline:
```bash
source venv/bin/activate
python run.py --rtsp "rtsp://localhost:8554/nest-cam"
```

### CLI Arguments for `run.py`

You can customize the daemon at startup using the following command-line flags:

* `--rtsp`: RTSP stream URL (overrides `CCTV_RTSP_URL`).
* `--tripwire`: Overrides the tripwire line segment. Accepts a JSON array `[[x1,y1],[x2,y2]]` or a comma-separated list of 4 floats `x1,y1,x2,y2`.
* `--roi`: Overrides the motion ROI polygon. Accepts a JSON array of coordinate pairs `[[x1,y1],[x2,y2],...]` or a comma-separated list of floats `x1,y1,x2,y2,...`.
* `--model`: YOLO model name or local path (overrides `CCTV_MODEL_NAME`).
* `--host`: Host to bind the FastAPI server to (overrides `CCTV_HOST`).
* `--port`: Port to bind the FastAPI server to (overrides `CCTV_PORT`).
* `--no-api`: Starts the stream-monitoring pipeline thread but disables the FastAPI web server.
* `--no-pipeline`: Starts the FastAPI web server but disables the background stream-monitoring pipeline.

### Event Webhooks

When a tripwire crossing event is detected, the pipeline automatically sends asynchronous `HTTP POST` requests to the configured `webhook_urls`.

The webhook payload has the following structure:
```json
{
  "event_id": 12,
  "event_type": "ENTER",
  "tracker_id": 3,
  "confidence": 0.94,
  "snapshot_path": "snapshots/enter_id3_20260526_173000_123456.jpg",
  "is_someone_home": true,
  "current_occupancy": 1,
  "timestamp": "2026-05-26T17:30:00.123456"
}
```

If any configured webhook fails (due to connection timeout, DNS resolution issues, or returning a non-2xx status code), the failure is caught and logged, allowing other webhooks and the main tracking pipeline to continue unimpeded.

### API Endpoints
* **Get Current Status**: `GET http://localhost:8000/status`
  * Returns: `{"is_someone_home": true, "current_occupancy": 1, "last_updated": "..."}`
* **Fetch Recent Events**: `GET http://localhost:8000/events?limit=10`
* **Manual Override/Reset**: `POST http://localhost:8000/reset`
  * Body: `{"is_someone_home": false, "current_occupancy": 0}`
* **Fetch Event Snapshots**: Files can be served directly from `/snapshots/...` (e.g., `http://localhost:8000/snapshots/enter_id9_20260522_210145_390123.jpg`).

---

## 🛠 Calibration & Debugging Utilities

The repository provides two OpenCV-based graphical utilities to help calibrate tripwires/regions of interest and visualize YOLO tracking behavior in real-time.

> [!NOTE]
> Since these tools use OpenCV GUI features (`cv2.imshow`), they must be run in a desktop environment or a system with X11 forwarding enabled. They will not function on headless environments without a display server.

### 1. Calibration Tool (`calibrate.py`)

[calibrate.py](calibrate.py) allows you to visually draw and configure the tripwire line segment or the motion detection Region of Interest (ROI) polygon directly on a camera stream frame.

#### Usage
To calibrate the **tripwire line**:
```bash
python calibrate.py --rtsp "rtsp://localhost:8554/nest-cam" --mode tripwire
```

To calibrate the **motion ROI polygon**:
```bash
python calibrate.py --rtsp "rtsp://localhost:8554/nest-cam" --mode roi
```

#### Controls
* **Left Click**: Place Point A and B (in `tripwire` mode) or vertices (in `roi` mode).
* **[Enter]**: Close the polygon (in `roi` mode, requires at least 3 vertices).
* **[S]**: Save the coordinates directly to `config.json` under `tripwire_line` or `motion_roi` as normalized coordinates.
* **[R]**: Reset the current drawing.
* **[Q] or [ESC]**: Quit the calibration tool without saving.

---

### 2. Tracker Visualization Utility (`visualize_tracker.py`)

[visualize_tracker.py](visualize_tracker.py) is a real-time visualization tool that displays the YOLO bounding boxes, active tracking IDs, centroid history paths, tripwire lines, dead zones, and motion ROI polygons. It includes a comprehensive sidebar HUD showing live performance metrics.

#### Usage
```bash
python visualize_tracker.py --rtsp "rtsp://localhost:8554/nest-cam"
```
You can also run it against offline video files (e.g., for testing and development):
```bash
python visualize_tracker.py --rtsp "path/to/test_video.mp4"
```

#### CLI Options
* `--rtsp`: RTSP stream URL or path to offline video file (defaults to `CCTV_RTSP_URL`).
* `--model`: YOLO model name or path to a local `.pt` weight file (defaults to `CCTV_MODEL_NAME`).
* `--conf`: Confidence threshold for detections (defaults to `tracker_confidence`).
* `--track-buffer`: Frame limit to keep lost tracks alive (defaults to `track_buffer`).
* `--config`: Path to the config JSON file to use or live-reload (defaults to `config.json`).

#### Real-time Keyboard Shortcuts
* **[P]**: Pause / Resume video playback.
* **[R]**: Reset the object tracker's active states and history trails.
* **[L]**: Live-reload parameters (like tripwire positions, thresholds) directly from `config.json` without restarting the script.
* **[M]**: Toggle visibility of the Motion ROI layer.
* **[T]**: Toggle visibility of the Tripwire & Dead Zone layers.
* **[H]**: Toggle visibility of the historical centroid trail layers.
* **[Q] or [ESC]**: Quit the utility.

---

## 💬 Mattermost Integration

To let users ask *"Is there anyone in the house?"* inside Mattermost:

1. Go to **Mattermost System Console > Integrations > Slash Commands**.
2. Create a new command (e.g. `/whoshome`).
3. Set the **Request URL** to `http://<your-linux-server-ip>:8000/status` (or use a reverse proxy like Nginx with Auth).
4. Implement a lightweight middleware/webhook handler or configure your command to parse the FastAPI response and output a friendly markdown message:
   ```markdown
   🟢 **Someone is home.**
   * **Current occupancy:** 1 occupant(s)
   * **Last transition:** Detected at 2026-05-22 21:01
   ```

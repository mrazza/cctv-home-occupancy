# House Presence Monitoring System (Nest Camera)

A private, high-performance local Linux pipeline to monitor presence in your home using Nest cameras. It operates in a **"Fast & Slow"** fashion to optimize CPU utilization:
1. **Fast Stage (0% CPU idle)**: Continuous, ultra-lightweight pixel differencing with OpenCV.
2. **Slow Stage (Active AI)**: Activates YOLOv8/11 and ByteTrack to track people crossing virtual tripwires only when motion is detected.

Exposes a FastAPI endpoint that can be directly queried or hooked into a Mattermost Chatbot Slash Command.

---

## How It Works: The "Fast & Slow" Pipeline

Continuous deep-learning inference (YOLO) is highly resource-intensive. To resolve this, this system utilizes a dual-stage pipeline:
* **IDLE State (Fast OpenCV Background Subtraction)**: The system decodes the stream and compares incoming frames using a lightweight weighted running average. This uses almost **0% CPU**.
* **ACTIVE State (Slow YOLO Tracking)**: When motion is detected above your configured threshold, the pipeline wakes up the YOLO object tracker. YOLO maps trajectories for 'person' classes and calculates line-crossings. After **150 frames (15 seconds)** of zero motion and zero active tracks, the system gracefully sleeps back into the **IDLE** state.

---

## Mathematical Tripwire (Line-Crossing) Logic

The tripwire is modeled as a **directed vector** from Point $A(x_1, y_1)$ to Point $B(x_2, y_2)$. 

When a person moves from previous centroid $P_{prev}$ to current centroid $P_{curr}$:
1. **Intersection Check**: The system checks if segment $AB$ and $P_{prev}P_{curr}$ intersect using counter-clockwise (CCW) orientation math via vector determinants.
2. **Direction Check (Cross-Product)**: If an intersection occurs, we calculate the sign of the cross-product of $ec{v}_{AB}$ with the displacement vector from $A$ to the current point $P_{curr}$:
   $$	ext{Side} = 	ext{sign}left((B_x - A_x) cdot (P_y - A_y) - (B_y - A_y) cdot (P_x - A_x)ight)$$
   * **Side = +1**: Left/Clockwise side of directed vector $AB$ (designated as **INSIDE**).
   * **Side = -1**: Right/Counter-clockwise side of directed vector $AB$ (designated as **OUTSIDE**).

Transitions are mapped as:
* **Outside (-1) $	o$ Inside (+1)** = `ENTER` 🟢 (Increments occupancy count)
* **Inside (+1) $	o$ Outside (-1)** = `LEAVE` 🔴 (Decrements occupancy count)

---

## Configuration & Calibration

All coordinates are **normalized between `0.0` and `1.0`** (independent of the camera's raw pixel resolution). This means if your RTSP stream switches resolution, your tripwire lines will not break.

### 1. Calibrating your Doorway Coordinates
1. Take a screenshot snapshot of your camera feed.
2. Map a coordinate plane from top-left `(0.0, 0.0)` to bottom-right `(1.0, 1.0)`.
3. Draw your line across the doorway:
   * **Point A (Start)**: `(x1, y1)`
   * **Point B (End)**: `(x2, y2)`
   * *Note*: Remember that the left side of your line vector ($A 	o B$) is considered **Inside**, and the right side is **Outside**. Ensure you draw it such that entering the home crosses from right to left!

### 2. Customizing Settings
You can customize the system configuration in `src/config.py`, or set them dynamically at runtime via **Environment Variables**:

| Env Variable | Description | Default |
| :--- | :--- | :--- |
| `CCTV_RTSP_URL` | RTSP stream from your Nest bridge (e.g. go2rtc/Scrypted) | `rtsp://localhost:8554/nest-cam` |
| `CCTV_FPS_LIMIT` | Framerate cap for processing | `10` |
| `CCTV_MOTION_THRESHOLD` | Pixel change sensitivity fraction (0.0 to 1.0) to wake YOLO | `0.005` (0.5%) |
| `CCTV_MIN_CONTOUR_AREA` | Minimum pixel blob size to count as motion | `500` |
| `CCTV_MOTION_COOLDOWN` | Number of frames with zero motion before going back to sleep | `150` |
| `CCTV_DB_PATH` | SQLite database file path | `db/presence.db` |
| `CCTV_SNAPSHOT_DIR` | Directory where face/body crops are saved | `snapshots/` |

---

## Installation & Setup

### 1. Prerequisites
You need a Nest Camera RTSP stream. Because Nest streams expire quickly and utilize WebRTC, it is highly recommended to run **go2rtc** or **Scrypted** in a local Docker container to bridge the stream to a persistent local RTSP url.

### 2. Set Up Python Environment
```bash
cd sources/mrazza/cctv-monitoring
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Suite of Tests
Run the unit and integration tests to verify the math, database, and API routing:
```bash
pytest
```

---

## Running the Application

To run the camera pipeline and API server concurrently on your Linux machine, execute:
```bash
python run.py --rtsp "rtsp://YOUR_LOCAL_BRIDGE_IP:8554/nest-cam"
```

### Main API Endpoints:
The daemon spawns a local FastAPI server (default: `http://0.0.0.0:8000`).
* **`GET /status`**: Query current presence state.
  ```bash
  curl http://localhost:8000/status
  # Returns: {"is_someone_home": true, "current_occupancy": 1, "last_updated": "2026-05-22T21:01:45"}
  ```
* **`GET /events`**: Query recent transitions.
* **`POST /reset`**: Force a manual sync (overriding the occupancy count if state falls out of sync).
  ```json
  {
    "is_someone_home": true,
    "current_occupancy": 1
  }
  ```
* **`GET /snapshot/{filename}`**: Access individual face/body cropped snapshots.

---

## Connecting to Mattermost

You can easily query your system from Mattermost by creating an **Outgoing Webhook** or a **Slash Command** (e.g. `/whoshome`):
1. Navigate to Mattermost **Product Menu > Integrations > Slash Commands**.
2. Create a command (e.g., `/whoshome`).
3. Set the Request URL to your Linux server's API: `http://YOUR_SERVER_IP:8000/status`.
4. Configure your bot to format the response into markdown:
   > 🟢 **Someone is currently in the house.** 
   > * Total Estimated Occupants: 1
   > * Last activity recorded: 9:02 PM.

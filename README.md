# House Presence Monitoring System (cctv-home-occupancy)

A private, high-performance local Linux pipeline to monitor presence in your home using Nest cameras. It operates in a **"Fast & Slow"** fashion to optimize CPU utilization:
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

## ⚙️ Environment Variables Config

You can customize runtime parameters using environment variables without editing the source code:

| Env Variable | Default Value | Description |
| :--- | :--- | :--- |
| `CCTV_RTSP_URL` | `rtsp://localhost:8554/nest-cam` | The local camera stream URL. |
| `CCTV_FPS_LIMIT` | `10` | Target frames per second to process. |
| `CCTV_MOTION_THRESHOLD` | `0.005` | Percent of changed pixels (0.0 to 1.0) to trigger motion state. |
| `CCTV_MIN_CONTOUR_AREA` | `500` | Minimum pixel area of a moving object. |
| `CCTV_MOTION_COOLDOWN` | `150` | How many frames of silence before shutting off YOLO. |
| `CCTV_DB_PATH` | `db/presence.db` | Path to the SQLite presence database. |
| `CCTV_SNAPSHOT_DIR` | `snapshots` | Folder path where face/body crops are stored. |

---

## 🏃 Running the Application

To start both the API server and the background camera-monitoring pipeline:
```bash
source venv/bin/activate
python run.py --rtsp "rtsp://localhost:8554/nest-cam"
```

### API Endpoints
* **Get Current Status**: `GET http://localhost:8000/status`
  * Returns: `{"is_someone_home": true, "current_occupancy": 1, "last_updated": "..."}`
* **Fetch Recent Events**: `GET http://localhost:8000/events?limit=10`
* **Manual Override/Reset**: `POST http://localhost:8000/reset`
  * Body: `{"is_someone_home": false, "current_occupancy": 0}`
* **Fetch Event Snapshots**: Files can be served directly from `/snapshots/...` (e.g., `http://localhost:8000/snapshots/enter_id9_20260522_210145_390123.jpg`).

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

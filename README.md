# House Presence Monitoring System (Nest Camera)

A private, high-performance local Linux pipeline to monitor presence in your home using Nest cameras. It operates in a **"Fast & Slow"** fashion to optimize CPU utilization:
1. **Fast Stage (0% CPU idle)**: Continuous, ultra-lightweight pixel differencing with OpenCV.
2. **Slow Stage (Active AI)**: Activates YOLOv8 and ByteTrack to track people crossing virtual tripwires only when motion is detected.

Exposes a FastAPI endpoint that can be directly queried or hooked into a Mattermost Chatbot Slash Command.

---

## Features
- **RTSP Input**: Integrated with local WebRTC-to-RTSP bridges (e.g. Scrypted or go2rtc).
- **CPU Optimized**: Uses background subtraction to wake/sleep the deep learning tracking logic.
- **Directional Crossing Math**: Uses geometric intersection and vector cross-product to detect entering vs. leaving.
- **Self-Correcting State Engine**: Maintains transactional SQLite DB representing household occupancy.
- **Queryable API**: FastAPI allows Mattermost bots to query "Is anyone home?" or fetch snapshots of recent entry/exit events.
- **Facial Crop Support**: Automatically isolates crops of faces/bodies for future facial identification.

---

## Installation & Setup

### 1. Prerequisites
- Linux Server (with Docker and Python 3.10+ installed).
- A Nest Camera connected to **Scrypted** or **go2rtc** exposing a local RTSP URL (e.g., `rtsp://localhost:8554/nest-cam`).

### 2. Install Dependencies
```bash
cd sources/mrazza/cctv-monitoring
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Unit and Integration Tests
To make sure all tracking math, state engines, and API endpoints function perfectly:
```bash
pytest
```

---

## Development & Structure
Please refer to [PLAN.md](PLAN.md) for architectural details, state machine logic, and the mathematical formulas for the line crossing checking.

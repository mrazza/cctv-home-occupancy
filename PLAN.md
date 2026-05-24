# CCTV Monitoring Plan: Fast/Slow House Presence Detection

This document outlines the detailed architecture, mathematical algorithms, and database design for the local Nest Camera Presence Monitoring System.

## 1. System Architecture

```
                  +--------------------------------+
                  |  Nest Cam (WebRTC/RTSP Bridge)  |
                  +---------------+----------------+
                                  |
                                  v Local RTSP Feed
                  +---------------+----------------+
                  |   Frame Grabber & Decelerator  |
                  +---------------+----------------+
                                  | Frame Stream (10 FPS)
                                  v
+---------------------------------+---------------------------------+
|                         Orchestrator Pipeline                     |
|                                                                   |
|       +-----------------------------------------------------+     |
|       |                     IDLE State                      |     |
|       |             (Fast Motion Detection)                 |     |
|       |  Check frame differences. If movement > threshold:  |     |
|       |  --> Transition to ACTIVE state.                    |     |
|       +--------------------------+--------------------------+     |
|                                  |                                |
|                                  v Transition                     |
|                                                                   |
|       +--------------------------+--------------------------+     |
|       |                    ACTIVE State                     |     |
|       |             (Slow YOLO tracking)                    |     |
|       |  Run Ultralytics YOLOv8/11 + ByteTrack.             |     |
|       |  Calculate line intersections and state events.     |     |
|       |  If idle for 150 frames: --> Transition to IDLE.    |     |
|       +-----------------------------------------------------+     |
+---------------------------------+---------------------------------+
                                  |
                                  v Event (ENTER / LEAVE)
                  +---------------+----------------+
                  |   Database & State Engine      | <---+ Manual Correction
                  |          (SQLite)              |
                  +---------------+----------------+
                                  |
                                  v DB Queries
                  +---------------+----------------+
                  |       FastAPI Query Server     |
                  +---------------+----------------+
                                  ^
                                  | Webhook
                  +---------------+----------------+
                  |        Mattermost Bot          |
                  +--------------------------------+
```

---

## 2. Fast Motion Detection (The Frame-Differencing Engine)

To save CPU cycles, the camera's feed is continuously parsed with a fast frame-differencing detector. 

### Algorithm Steps:
1. **Grayscale & Gaussian Blur**: Convert the frame to grayscale and apply Gaussian blur to eliminate high-frequency noise:
   $$F_{blur} = \text{GaussianBlur}(\text{Grayscale}(F), \text{kernel\_size}=(21, 21), \text{sigma}=0)$$
2. **Background Accumulation / Absolute Difference**: 
   Maintain an accumulated/running average frame $B$ representing the background. When the current frame $F_{blur}$ arrives:
   $$D = |F_{blur} - B|$$
3. **Thresholding**:
   Apply binary thresholding to isolate significant pixel changes:
   $$T = \text{Threshold}(D, \text{thresh}=25, \text{maxval}=255)$$
4. **Dilation**:
   Dilate the thresholded image to merge adjacent small spots of motion:
   $$T_{dilated} = \text{Dilate}(T, \text{iterations}=2)$$
5. **Contour Analysis**:
   Find contours on $T_{dilated}$. Calculate the area of all contours:
   $$\text{Total Motion Area} = \sum_{c \in \text{Contours}} \text{Area}(c)$$
   If $\text{Total Motion Area} > \text{Threshold}$, mark `motion_detected = True`.

---

## 3. Slow Object Detection & Tripwire Tracking (The AI Engine)

When `motion_detected = True`, the system spawns the YOLOv8/11 object tracker.

### The Math of Line-Crossing (Tripwires)

We define a **directed line segment** representing the threshold/doorway of the house:
- **Start point**: $A(x_1, y_1)$ (Inside or Boundary edge)
- **End point**: $B(x_2, y_2)$

For any tracked person, their trajectory is modeled by their consecutive position points. Let's look at their movement from previous point $P_{prev}(x_{p1}, y_{p1})$ to current point $P_{curr}(x_{p2}, y_{p2})$.

#### Intersection Check:
Two segments $AB$ and $CD$ (where $C = P_{prev}$ and $D = P_{curr}$) intersect if and only if the orientation of the triplets $(A, B, C)$ and $(A, B, D)$ are different, and the orientation of $(C, D, A)$ and $(C, D, B)$ are different.

Orientation of triplet $(X, Y, Z)$ is calculated using the determinant of their vectors:
$$\text{val} = (Y_y - X_y) \cdot (Z_x - Y_x) - (Y_x - X_x) \cdot (Z_y - Y_y)$$
- $\text{val} = 0$: Collinear
- $\text{val} > 0$: Clockwise
- $\text{val} < 0$: Counterclockwise

#### Direction Check (Cross-Product):
Once we confirm the trajectory $CD$ intersects our tripwire $AB$:
We calculate the cross product of the tripwire vector $\vec{v}_{AB} = B - A$ and the movement vector $\vec{v}_{CD} = D - C$.
Alternatively, we can use the sign of the cross product of $\vec{v}_{AB}$ with the displacement vector from $A$ to the final position $D$:
$$\text{Direction} = \text{sign}\left((B_x - A_x) \cdot (D_y - A_y) - (B_y - A_y) \cdot (D_x - A_x)\right)$$
- If **positive (+1)**: The person crossed the line to the **right/clockwise side** (designated as **ENTER**).
- If **negative (-1)**: The person crossed the line to the **left/counter-clockwise side** (designated as **LEAVE**).

---

## 4. State & Database Design

### SQLite Schema (`db/presence.db`)

#### Table: `presence_state`
Tracks the current state of occupancy in the house.
```sql
CREATE TABLE IF NOT EXISTS presence_state (
    id INTEGER PRIMARY KEY CHECK (id = 1), -- Enforces single row
    is_someone_home BOOLEAN NOT NULL DEFAULT 0,
    current_occupancy INTEGER NOT NULL DEFAULT 0,
    last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

#### Table: `events_log`
Audit log of all detected transitions.
```sql
CREATE TABLE IF NOT EXISTS events_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL, -- 'ENTER', 'LEAVE', 'FORCE_RESET'
    tracker_id INTEGER, -- YOLO track ID
    confidence REAL, -- Detection confidence
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    snapshot_path TEXT -- Path to saved crop of person
);
```

---

## 5. Directory Structure

```
cctv-home-occupancy/
├── PLAN.md                   # Full architecture and mathematical spec
├── README.md                 # Setup and run guide
├── pyproject.toml            # Poetry / packaging configuration
├── requirements.txt          # Python dependency specifications
├── pytest.ini                # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── config.py             # App-wide settings and coordinate inputs
│   ├── database.py           # SQLite database layer
│   ├── motion_detector.py    # OpenCV fast motion detection
│   ├── object_tracker.py     # YOLO + ByteTrack and tripwire math
│   ├── pipeline.py           # Orchestrator (Fast/Slow switcher)
│   └── api.py                # FastAPI endpoints
└── tests/
    ├── __init__.py
    ├── conftest.py           # Test fixtures and database mock setup
    ├── test_database.py      # Unit tests for DB layer
    ├── test_motion_detector.py # Unit tests for motion detection
    ├── test_object_tracker.py  # Unit/math tests for tracking and crossing
    ├── test_pipeline.py      # Integration tests for orchestration
    └── test_api.py           # Unit tests for query endpoints
```

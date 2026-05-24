# 🔌 Scrypted On-Demand Hybrid Streaming Guide

When monitoring Google Nest cameras, pulling a continuous RTSP stream (24/7) will quickly consume your Google API limits and trigger severe throttling or account bans. 

This system supports a **Hybrid On-Demand Streaming** mode. The camera stream is completely closed (0% bandwidth and 0% Google API usage) during idle times. When movement is detected, Scrypted triggers our Python pipeline, which temporarily spins up the stream, performs high-precision YOLOv8/ByteTrack tracking, and shuts the stream down as soon as the activity is over.

---

## 🏗️ Architecture & Flow

```
┌─────────────┐               ┌──────────┐               ┌────────────┐
│ Google Nest │  (1) Motion  │ Scrypted │ (2) Webhook   │ Python API │
│   Camera    ├──────────────>│ SDM App  ├──────────────>│  Endpoint  │
└─────────────┘               └────┬─────┘               └─────┬──────┘
                                   ▲                           │
                                   │                           │ (3) Lazily Start
                                   │ (4) Pull Stream           │     ThreadedReader
                                   │                           ▼
                                ┌──┴───────────────────────────┴──┐
                                │       PipelineOrchestrator      │
                                │    (YOLO / Tripwire Tracking)   │
                                └─────────────────────────────────┘
```

1. **Detection**: Google Nest Camera detects movement locally and fires a GCP Pub/Sub message.
2. **Notification**: Scrypted (acting as your smart home bridge) receives this event and instantly triggers our Python Webhook at `POST /webhook/trigger_motion`.
3. **Activation**: Our Python daemon lazily spins up the `ThreadedVideoReader` background thread and requests the camera's RTSP stream from Scrypted.
4. **Negotiation**: Connecting to Scrypted's RTSP rebroadcast URL triggers Scrypted to open the active live feed to Google Nest.
5. **Real-time Tracking**: The frame pipeline processes frames at your target FPS limit (e.g. 10 FPS), analyzing tripwire crossings.
6. **Graceful Sleep**: Once the motion trigger window expires (default 45 seconds) and there are no active YOLO tracks on-screen, the orchestrator shuts down the `ThreadedVideoReader`.
7. **Stream Release**: When our Python reader disconnects, Scrypted notices there are `0` active clients watching the feed and immediately terminates the Nest live stream, saving your API quota!

---

## ⚙️ Step-by-Step Setup Guide

### 1. Configure the Python Daemon

To set up the service to run in event-driven mode instead of continuous streaming, modify your configuration:

#### Option A: `config.json`
Update or create `config.json` in your project root:
```json
{
  "trigger_mode": "event",
  "event_stream_duration": 45
}
```

#### Option B: Environment Variables
If running via Docker or systemd, set the following environment variables:
```bash
export CCTV_TRIGGER_MODE="event"
export CCTV_EVENT_STREAM_DURATION="45"
```

Start your Python daemon as usual:
```bash
python run.py
```
*Note: You will see a log line stating: `Initializing stream reader with trigger mode: 'event'`.*

---

### 2. Configure Scrypted Webhook

You need to tell Scrypted to hit your daemon's webhook when motion is detected.

#### Method A: Using Scrypted Automation / Scripting (Recommended)
1. Open your **Scrypted Management Console**.
2. Click on **Automations** in the left sidebar (or install the **Automation** / **Scripting** plugin if not already installed).
3. Create a new Automation Rule:
   - **Name**: `Trigger Tripwire Stream`
   - **Trigger**: Select your Nest Camera and choose **Motion Detected**.
   - **Action**: Add a Javascript/Script action with the following snippet:
     ```javascript
     // Replace with your python API host and port
     const API_URL = "http://192.168.1.100:8000/webhook/trigger_motion";
     
     log.info("Sending trigger to presence tripwire daemon...");
     sdk.systemManager.getDeviceByName("HttpClient")
        .post(API_URL, null)
        .then(() => log.info("Successfully triggered daemon!"))
        .catch(err => log.error("Failed to trigger tripwire daemon: " + err));
     ```

#### Method B: Using Scrypted Webhooks Plugin
1. Install the **Webhooks** plugin in Scrypted.
2. Set up a Webhook Extension on your Nest Camera.
3. Configure a Rule:
   - When **Motion Detected** on `Nest Camera`.
   - Send `POST` to `http://<YOUR_DAEMON_IP>:8000/webhook/trigger_motion`.

---

## 🔍 Verification & Diagnostics

Once set up, you can monitor the daemon's logs to verify it is working as expected:

1. **Idle State**:
   ```
   [2026-05-24 12:00:00] [INFO] [MainThread] [src.pipeline]: Initializing stream reader with trigger mode: 'event' for: rtsp://192.168.1.200:8554/nest-cam
   [2026-05-24 12:00:00] [INFO] [MainThread] [src.pipeline]: Stream pipeline started with delay of 0.100s between checks.
   ```
   *(No RTSP connections are open here)*

2. **When Motion Occurs**:
   When Scrypted sends the webhook, you'll see:
   ```
   [2026-05-24 12:00:15] [INFO] [MainThread] [src.pipeline]: On-demand event-driven window triggered/extended until epoch 1779624060.0 (for 45s)
   [2026-05-24 12:00:15] [INFO] [MainThread] [src.pipeline]: Trigger window active. Lazily starting ThreadedVideoReader...
   [2026-05-24 12:00:16] [INFO] [ThreadedVideoReader] [src.pipeline]: ThreadedVideoReader connection initialized successfully.
   [2026-05-24 12:00:18] [INFO] [MainThread] [src.pipeline]: Heartbeat: ThreadedVideoReader has successfully processed 3 frames.
   ```

3. **Going Back to Sleep**:
   Once 45 seconds pass without further motion triggers or active tracks:
   ```
   [2026-05-24 12:01:00] [INFO] [MainThread] [src.pipeline]: Trigger window expired and no active tracks. Stopping ThreadedVideoReader to release RTSP quota...
   [2026-05-24 12:01:00] [INFO] [MainThread] [src.pipeline]: Stopping ThreadedVideoReader...
   [2026-05-24 12:01:00] [INFO] [MainThread] [src.pipeline]: ThreadedVideoReader stopped.
   ```

---

## 💡 Troubleshooting Latency

Because starting a stream has some handshake overhead (~2-3 seconds), follow these best practices:
- **Broad Camera Angle**: Ensure your camera covers an area where people walk for a few seconds *before* they cross the tripwire (e.g., driveway, long porch). This naturally masks the stream initialization delay.
- **RTSP Rebroadcast Plugin**: Make sure Scrypted is configured with the **RTSP Rebroadcast** plugin. It optimizes streaming and multiplexes RTSP feeds with low overhead.

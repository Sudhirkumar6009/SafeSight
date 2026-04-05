# ViolenceSense RTSP Live Stream Service

Real-time RTSP stream ingestion and violence detection service for ViolenceSense.

## 🚀 Features

- **RTSP/RTMP Stream Ingestion** - FFmpeg/OpenCV-based stream capture
- **Sliding Window Inference** - Continuous 2-4 second window analysis
- **Event Detection Logic** - Threshold + duration rules for real alerts
- **Clip Recording** - Automatic clip saving with before/after context
- **Alert Dashboard** - Web UI for stream monitoring and clip review
- **SQLite Database** - Local event storage with confirm/dismiss workflow
- **WebSocket Updates** - Real-time score and alert notifications

## 📋 Prerequisites

### Required Software

1. **Python 3.10+**
2. **FFmpeg** (for RTSP streaming)

   ```bash
   # Windows (using Chocolatey)
   choco install ffmpeg

   # Windows (using Scoop)
   scoop install ffmpeg

   # Linux
   sudo apt install ffmpeg

   # macOS
   brew install ffmpeg
   ```

3. **MediaMTX** (optional, for creating test RTSP streams)
   - Download from: https://github.com/bluenviron/mediamtx/releases

## 🔧 Installation

### 1. Create Virtual Environment

```bash
cd rtsp-service
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example config
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS

# Edit .env with your settings
```

### 4. Start the Service

```bash
python main.py
```

The service will start on `http://localhost:8080`

## 📹 Stream Sources

### 1. Real CCTV Cameras (RTSP)

IP cameras that expose RTSP URLs:

```
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# CP Plus
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1

# Axis
rtsp://admin:password@192.168.1.100:554/axis-media/media.amp

# Generic ONVIF
rtsp://admin:password@192.168.1.100:554/stream1
```

### 2. Convert MP4 to RTSP (Development/Demos)

Use FFmpeg to stream a video file as RTSP:

```bash
# First, start MediaMTX RTSP server
mediamtx

# In another terminal, stream MP4 to RTSP
ffmpeg -re -stream_loop -1 -i video.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp -rtsp_transport tcp rtsp://localhost:8554/stream

# Or use the included tool
python tools/rtsp_server.py --mode mp4 --file video.mp4 --loop
```

The stream will be available at: `rtsp://localhost:8554/stream`

### 3. Local Webcam as RTSP

Stream your webcam via RTSP:

```bash
# Using the included tool
python tools/rtsp_server.py --mode webcam --camera 0

# Or directly with FFmpeg (Windows)
ffmpeg -f dshow -i video="Your Webcam Name" -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/webcam
```

### 4. Public RTSP Test Streams

For pipeline testing only (may be unstable):

```bash
# List available test streams
python tools/rtsp_server.py --mode test
```

Example streams:

- `rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4`

## 🖥️ Usage

### Start the Service

```bash
python main.py
```

### Open the Dashboard

Open `dashboard/index.html` in your browser, or:

```bash
# Serve with Python
cd dashboard
python -m http.server 8000
# Open http://localhost:8000
```

### API Endpoints

| Endpoint                      | Method    | Description        |
| ----------------------------- | --------- | ------------------ |
| `/api/v1/health`              | GET       | Health check       |
| `/api/v1/streams`             | GET       | List all streams   |
| `/api/v1/streams`             | POST      | Add a new stream   |
| `/api/v1/streams/{id}/start`  | POST      | Start a stream     |
| `/api/v1/streams/{id}/stop`   | POST      | Stop a stream      |
| `/api/v1/streams/{id}`        | DELETE    | Remove a stream    |
| `/api/v1/events`              | GET       | List events        |
| `/api/v1/events/pending`      | GET       | Get pending alerts |
| `/api/v1/events/{id}/confirm` | POST      | Confirm an event   |
| `/api/v1/events/{id}/dismiss` | POST      | Dismiss an event   |
| `/api/v1/clips/{filename}`    | GET       | Download a clip    |
| `/api/v1/ws`                  | WebSocket | Real-time updates  |

### Add a Stream via API

```bash
curl -X POST http://localhost:8080/api/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Front Entrance",
    "url": "rtsp://admin:password@192.168.1.100:554/stream",
    "stream_type": "rtsp",
    "location": "Building A",
    "auto_start": true
  }'
```

### Add Webcam Stream

```bash
curl -X POST http://localhost:8080/api/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Local Webcam",
    "url": "0",
    "stream_type": "webcam",
    "auto_start": true
  }'
```

## ⚙️ Configuration

### Environment Variables

| Variable                 | Default                                       | Description                  |
| ------------------------ | --------------------------------------------- | ---------------------------- |
| `HOST`                   | 0.0.0.0                                       | Server host                  |
| `PORT`                   | 8080                                          | Server port                  |
| `ML_SERVICE_URL`         | http://localhost:8000                         | ML service URL               |
| `SLIDING_WINDOW_SECONDS` | 3                                             | Inference window size        |
| `FRAME_SAMPLE_RATE`      | 8                                             | Frames per inference         |
| `VIOLENCE_THRESHOLD`     | 0.65                                          | Violence detection threshold |
| `MIN_CONSECUTIVE_FRAMES` | 5                                             | Frames before alert          |
| `ALERT_COOLDOWN_SECONDS` | 10                                            | Cooldown between alerts      |
| `CLIP_DURATION_BEFORE`   | 5                                             | Seconds before event         |
| `CLIP_DURATION_AFTER`    | 10                                            | Seconds after event          |
| `CLIPS_DIR`              | ./clips                                       | Clip storage directory       |
| `DATABASE_URL`           | postgresql+asyncpg://...                      | Async PostgreSQL connection  |
| `MODEL_PATH`             | ../ml-service/models/violence_model_legacy.h5 | Local model path             |

## 📊 Event Flow

```
Stream Frame → Frame Buffer → Sliding Window → Inference → Score
                                                            ↓
                                              Threshold Check (0.65)
                                                            ↓
                                              Duration Check (5 frames)
                                                            ↓
                                              Create Event + Record Clip
                                                            ↓
                                              WebSocket Alert → Dashboard
                                                            ↓
                                              Review (Confirm/Dismiss)
```

## 🎯 Event Detection Logic

1. **Threshold**: Violence score ≥ 0.65 (configurable)
2. **Duration**: 5+ consecutive high-scoring frames
3. **Cooldown**: 10 seconds between alerts (prevents spam)
4. **Clip Recording**: 5s before + event duration + 10s after

## 📁 Project Structure

```
rtsp-service/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env                    # Configuration
├── app/
│   ├── config.py          # Settings management
│   ├── database.py        # SQLAlchemy models
│   ├── manager.py         # Stream manager
│   ├── api/
│   │   └── routes.py      # REST API endpoints
│   ├── stream/
│   │   └── ingestion.py   # RTSP ingestion
│   ├── inference/
│   │   └── pipeline.py    # ML inference
│   └── events/
│       └── detector.py    # Event detection
├── dashboard/
│   └── index.html         # Web dashboard
├── tools/
│   └── rtsp_server.py     # Test stream utilities
├── clips/                  # Recorded clips
└── logs/                   # Application logs
```

## 🔌 WebSocket Events

Connect to `ws://localhost:8080/api/v1/ws` for real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8080/api/v1/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "inference_result":
      // Real-time violence scores
      console.log(`Stream ${data.stream_id}: ${data.violence_score}`);
      break;

    case "event_start":
      // New violence event detected
      console.log(`Alert! Event ${data.event_id} started`);
      break;

    case "event_end":
      // Event ended, clip available
      console.log(`Event ${data.event_id} ended, clip: ${data.clip_path}`);
      break;

    case "stream_status":
      // Stream connected/disconnected
      console.log(`Stream ${data.stream_id}: ${data.status}`);
      break;
  }
};
```

## 🧪 Testing

### Run Demo with Webcam

```bash
# Start the service
python main.py

# In another terminal, add demo webcam stream
curl -X POST http://localhost:8080/api/v1/test/add-demo-stream
```

### Simulate an Event

```bash
curl -X POST "http://localhost:8080/api/v1/test/simulate-event?stream_id=1"
```

### Run Unit Tests

```bash
pytest tests/
```

## 🔗 Integration with Main ViolenceSense

This service runs independently for local development. To integrate with the main ViolenceSense system:

1. Point `ML_SERVICE_URL` to your ML service
2. Events and clips are stored locally in SQLite
3. Use the API to sync events to your main database if needed

## 📝 License

MIT License - See LICENSE file in the main project.

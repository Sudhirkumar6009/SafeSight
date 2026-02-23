"""
Simple RTSP Stream Service with Violence Detection
===================================================
Minimal FastAPI application for RTSP stream playback with real-time ML inference.
"""

import sys
import os
import asyncio
import threading
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import deque
from uuid import uuid4

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


# ============== Model Configuration ==============
MODEL_PATH = Path(__file__).parent / ".." / "ml-service" / "models" / "violence_model_legacy.h5"
EXPECTED_FRAMES = 20  # Model expects 20 frames (based on saved model config)
TARGET_SIZE = (224, 224)
VGG16_OUTPUT_SHAPE = (7, 7, 512)  # VGG16 conv5 output shape for 224x224 input
INFERENCE_INTERVAL = 0.2  # Run inference every 0.2 seconds
VIOLENCE_THRESHOLD = float(os.getenv("VIOLENCE_THRESHOLD", "0.50"))  # 50% for is_violent flag
VIOLENCE_ALERT_THRESHOLD = float(os.getenv("VIOLENCE_ALERT_THRESHOLD", "0.90"))  # Alert at 90%+ (instant notification)
VIOLENCE_ALERT_COOLDOWN = float(os.getenv("VIOLENCE_ALERT_COOLDOWN", "5.0"))  # 5 second cooldown between alerts


# ============== Violence Detection Model ==============

class ViolenceDetector:
    """
    Loads and runs the violence detection model.
    
    The model architecture expects VGG16 pre-extracted features:
    - Input: (batch, 20 frames, 7, 7, 512) - VGG16 conv5 block features
    - TimeDistributed(Flatten): (batch, 20, 25088)
    - LSTM(256): (batch, 256)
    - Dense(1, sigmoid): violence probability
    
    VGG16 is used as a separate feature extractor.
    """
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False
        self._lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model and VGG16 feature extractor."""
        try:
            model_path = MODEL_PATH.resolve()
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}")
                return
            
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
            
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            
            # 1. Load VGG16 as feature extractor (pre-trained on ImageNet)
            logger.info("Loading VGG16 feature extractor...")
            vgg16_base = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            vgg16_base.trainable = False  # Freeze weights
            self.feature_extractor = vgg16_base
            logger.info(f"✅ VGG16 loaded, output shape: {vgg16_base.output_shape}")
            
            # 2. Build the LSTM model architecture matching the saved weights
            # The saved model expects input: (batch, 20, 7, 7, 512)
            logger.info("Building LSTM classifier model...")
            lstm_input = keras.Input(shape=(EXPECTED_FRAMES, *VGG16_OUTPUT_SHAPE))
            x = layers.TimeDistributed(layers.Flatten())(lstm_input)
            x = layers.LSTM(256)(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            self.model = keras.Model(inputs=lstm_input, outputs=outputs)
            
            # 3. Load weights from the H5 file
            logger.info(f"Loading LSTM weights from {model_path}...")
            self.model.load_weights(str(model_path))
            self.is_loaded = True
            logger.info(f"✅ Loaded violence detection model weights")
            
            # 4. Warmup both models
            logger.info("Warming up models...")
            dummy_frame = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = self.feature_extractor.predict(dummy_frame, verbose=0)
            
            dummy_features = np.zeros((1, EXPECTED_FRAMES, *VGG16_OUTPUT_SHAPE), dtype=np.float32)
            _ = self.model.predict(dummy_features, verbose=0)
            logger.info("✅ Model warmup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
    def predict(self, frames: List[np.ndarray]) -> Optional[dict]:
        """Run prediction on frames."""
        if not self.is_loaded or self.model is None or self.feature_extractor is None:
            return None
        
        with self._lock:
            try:
                # Extract VGG16 features from frames
                features = self._extract_features(frames)
                
                # Run LSTM inference
                start = time.time()
                prediction = self.model.predict(features, verbose=0)
                inference_time = (time.time() - start) * 1000
                
                violence_score = float(prediction[0][0])
                
                return {
                    "violence_score": violence_score,
                    "non_violence_score": 1.0 - violence_score,
                    "is_violent": violence_score >= VIOLENCE_THRESHOLD,
                    "inference_time_ms": inference_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def _extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract VGG16 features from frames.
        
        Input: List of BGR frames (OpenCV format)
        Output: (1, num_frames, 7, 7, 512) - VGG16 conv5 features
        """
        from tensorflow.keras.applications.vgg16 import preprocess_input
        
        # Ensure we have exactly EXPECTED_FRAMES frames
        if len(frames) < EXPECTED_FRAMES:
            frames = list(frames) + [frames[-1]] * (EXPECTED_FRAMES - len(frames))
        elif len(frames) > EXPECTED_FRAMES:
            indices = np.linspace(0, len(frames) - 1, EXPECTED_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Preprocess frames for VGG16
        processed = []
        for frame in frames:
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to 224x224
            resized = cv2.resize(rgb_frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            # Convert to float
            img = resized.astype(np.float32)
            processed.append(img)
        
        # Stack frames: (num_frames, 224, 224, 3)
        batch = np.stack(processed, axis=0)
        
        # Apply VGG16 preprocessing (subtracts ImageNet mean, BGR conversion handled internally)
        batch = preprocess_input(batch)
        
        # Extract features: (num_frames, 7, 7, 512)
        features = self.feature_extractor.predict(batch, verbose=0)
        
        # Add batch dimension: (1, num_frames, 7, 7, 512)
        return np.expand_dims(features, axis=0)


# Global detector instance
detector = ViolenceDetector()


# ============== Simple Stream Class ==============

@dataclass
class StreamInfo:
    id: int
    name: str
    url: str
    is_running: bool = False
    is_connected: bool = False
    frame_count: int = 0
    error: Optional[str] = None


class SimpleRTSPStream:
    """RTSP stream handler with violence detection."""
    
    def __init__(self, stream_id: int, name: str, url: str):
        self.id = stream_id
        self.name = name
        self.url = url
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_connected = False
        self.frame_count = 0
        self.last_frame: Optional[np.ndarray] = None
        self.error: Optional[str] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Frame buffer for inference (sliding window)
        self.frame_buffer: deque = deque(maxlen=60)  # ~2 seconds at 30fps
        
        # Latest prediction
        self.last_prediction: Optional[dict] = None
        self.prediction_callback = None  # Set by manager
        
        # Violence alert cooldown tracking
        self._last_violence_alert_time = 0.0
    
    def start(self):
        """Start the stream capture and inference."""
        if self.is_running:
            return
        
        self.is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        # Start inference thread
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        
        logger.info(f"Started stream with inference: {self.name} ({self.url})")
    
    def stop(self):
        """Stop the stream capture."""
        self.is_running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._inference_thread:
            self._inference_thread.join(timeout=2)
        if self.capture:
            self.capture.release()
            self.capture = None
        self.is_connected = False
        logger.info(f"Stopped stream: {self.name}")
    
    def _capture_loop(self):
        """Main capture loop running in background thread."""
        while self.is_running:
            try:
                if self.capture is None or not self.capture.isOpened():
                    self._connect()
                    continue
                
                ret, frame = self.capture.read()
                if ret:
                    with self._lock:
                        self.last_frame = frame
                        self.frame_buffer.append(frame.copy())
                        self.frame_count += 1
                        self.is_connected = True
                        self.error = None
                else:
                    self.is_connected = False
                    self._connect()  # Try reconnecting
                    
            except Exception as e:
                self.error = str(e)
                self.is_connected = False
                logger.error(f"Stream error ({self.name}): {e}")
                time.sleep(1)
    
    def _inference_loop(self):
        """Run inference periodically on buffered frames."""
        while self.is_running:
            try:
                time.sleep(INFERENCE_INTERVAL)
                
                if not self.is_connected or not detector.is_loaded:
                    continue
                
                # Get frames from buffer
                with self._lock:
                    if len(self.frame_buffer) < EXPECTED_FRAMES // 2:
                        continue
                    frames = list(self.frame_buffer)
                
                # Run prediction
                result = detector.predict(frames)
                
                if result:
                    result["stream_id"] = str(self.id)
                    result["stream_name"] = self.name
                    self.last_prediction = result
                    
                    # Log result
                    score = result["violence_score"]
                    if score >= VIOLENCE_THRESHOLD:
                        logger.warning(f"🔴 VIOLENCE [{self.name}] score={score:.1%}")
                    else:
                        # Log every 5th normal result
                        if self.frame_count % 150 == 0:  # ~5 seconds
                            logger.info(f"📊 [{self.name}] score={score:.1%}")
                    
                    # Trigger callback for WebSocket broadcast
                    if self.prediction_callback:
                        self.prediction_callback(result)
                    
                    # Check if we should emit a violence alert (score >= 75%)
                    self._maybe_emit_violence_alert(result)
                        
            except Exception as e:
                logger.error(f"Inference error ({self.name}): {e}")
    
    def _maybe_emit_violence_alert(self, prediction: dict):
        """Emit a violence_alert WebSocket message when score exceeds threshold."""
        try:
            score = float(prediction.get("violence_score", 0.0))
        except (TypeError, ValueError):
            return
        
        # Only alert if score is above alert threshold (90%+ for instant alerts)
        if score < VIOLENCE_ALERT_THRESHOLD:
            return
        
        # Cooldown: don't spam alerts
        now = time.time()
        if (now - self._last_violence_alert_time) < VIOLENCE_ALERT_COOLDOWN:
            return
        
        self._last_violence_alert_time = now
        
        # Determine severity based on score
        if score >= 0.90:
            severity = "critical"
        elif score >= 0.80:
            severity = "high"
        else:
            severity = "medium"
        
        alert = {
            "type": "violence_alert",
            "event_id": str(uuid4()),
            "stream_id": str(self.id),
            "stream_name": self.name,
            "timestamp": prediction.get("timestamp") or datetime.utcnow().isoformat(),
            "confidence": score,
            "max_score": score,
            "max_confidence": score,
            "severity": severity,
            "message": f"Violence detected on {self.name} ({score * 100:.0f}% confidence)",
        }
        
        logger.warning(f"🚨 VIOLENCE ALERT: Violence detected on {self.name} - {score:.0%} confidence")
        broadcast_violence_alert(alert)
    
    def _connect(self):
        """Connect to the RTSP stream."""
        try:
            if self.capture:
                self.capture.release()
            
            logger.info(f"Connecting to: {self.url}")
            self.capture = cv2.VideoCapture(self.url)
            
            # Set buffer size to reduce latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.capture.isOpened():
                self.is_connected = True
                logger.info(f"Connected to: {self.name}")
            else:
                self.is_connected = False
                self.error = "Failed to open stream"
                import time
                time.sleep(2)  # Wait before retry
                
        except Exception as e:
            self.error = str(e)
            self.is_connected = False
            import time
            time.sleep(2)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        with self._lock:
            return self.last_frame.copy() if self.last_frame is not None else None
    
    def get_jpeg(self) -> Optional[bytes]:
        """Get the latest frame as JPEG bytes."""
        frame = self.get_frame()
        if frame is None:
            return None
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()
    
    def get_status(self) -> dict:
        """Get stream status in frontend-compatible format."""
        # Determine status string
        if self.is_running and self.is_connected:
            status = "running"
        elif self.is_running and not self.is_connected:
            status = "connecting"
        elif self.error:
            status = "error"
        else:
            status = "stopped"
        
        return {
            "id": str(self.id),  # Frontend expects string ID
            "name": self.name,
            "url": self.url,
            "rtsp_url": self.url,  # Frontend also uses rtsp_url
            "stream_type": "rtsp",
            "is_active": True,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "status": status,
            "frame_count": self.frame_count,
            "error_message": self.error,
            "inference_enabled": detector.is_loaded,
            "last_prediction": self.last_prediction
        }


# ============== Stream Manager ==============

# Global list for WebSocket connections (defined early for callback access)
active_connections: List[WebSocket] = []
main_event_loop = None  # Will store the main event loop reference


def _broadcast_ws(message_type: str, payload: dict):
    """Broadcast any WebSocket message to all connected clients."""
    global main_event_loop
    if not main_event_loop or not active_connections:
        return
    
    import json
    message = json.dumps({"type": message_type, "data": payload})
    
    for ws in active_connections[:]:
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(message), main_event_loop)
        except Exception:
            pass


def broadcast_prediction(prediction: dict):
    """Broadcast inference score to all WebSocket clients."""
    _broadcast_ws("inference_score", prediction)


def broadcast_violence_alert(alert: dict):
    """Broadcast violence alert notification to all WebSocket clients."""
    _broadcast_ws("violence_alert", alert)


class StreamManager:
    """Manages multiple RTSP streams with violence detection."""
    
    def __init__(self):
        self.streams: Dict[int, SimpleRTSPStream] = {}
        self._next_id = 1
    
    def add_stream(self, name: str, url: str, auto_start: bool = True) -> int:
        """Add a new stream."""
        stream_id = self._next_id
        self._next_id += 1
        
        stream = SimpleRTSPStream(stream_id, name, url)
        stream.prediction_callback = broadcast_prediction  # Set callback
        self.streams[stream_id] = stream
        
        if auto_start:
            stream.start()
        
        return stream_id
    
    def remove_stream(self, stream_id: int):
        """Remove a stream."""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            del self.streams[stream_id]
    
    def start_stream(self, stream_id: int):
        """Start a stream."""
        if stream_id in self.streams:
            self.streams[stream_id].start()
    
    def stop_stream(self, stream_id: int):
        """Stop a stream."""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
    
    def get_stream(self, stream_id: int) -> Optional[SimpleRTSPStream]:
        """Get a stream by ID (handles both int and string)."""
        if isinstance(stream_id, str):
            stream_id = int(stream_id)
        return self.streams.get(stream_id)
    
    def list_streams(self) -> List[dict]:
        """List all streams with status."""
        return [s.get_status() for s in self.streams.values()]
    
    def shutdown(self):
        """Stop all streams."""
        for stream in self.streams.values():
            stream.stop()


# Global stream manager
stream_manager = StreamManager()


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    logger.info("🚀 Starting Simple RTSP Service...")
    yield
    logger.info("🛑 Shutting down...")
    stream_manager.shutdown()


app = FastAPI(
    title="Simple RTSP Stream Service",
    description="Minimal RTSP stream playback service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== API Models ==============

class StreamCreate(BaseModel):
    name: str = Field(..., description="Stream name")
    url: str = Field(..., description="RTSP URL")
    auto_start: bool = Field(default=True, description="Auto-start stream")


# ============== API Routes ==============

@app.get("/")
async def root():
    return {"service": "Simple RTSP Stream Service", "status": "running"}


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "streams_count": len(stream_manager.streams)}


@app.post("/api/v1/streams")
async def create_stream(request: StreamCreate):
    """Add a new RTSP stream."""
    stream_id = stream_manager.add_stream(
        name=request.name,
        url=request.url,
        auto_start=request.auto_start
    )
    return {"success": True, "stream_id": str(stream_id)}


@app.get("/api/v1/streams")
async def list_streams():
    """List all streams."""
    return {"success": True, "data": stream_manager.list_streams()}


@app.get("/api/v1/streams/{stream_id}")
async def get_stream(stream_id: int):
    """Get stream status."""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    return {"success": True, "data": stream.get_status()}


@app.post("/api/v1/streams/{stream_id}/start")
async def start_stream(stream_id: int):
    """Start a stream."""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    stream.start()
    return {"success": True, "message": f"Stream {stream_id} started"}


@app.post("/api/v1/streams/{stream_id}/stop")
async def stop_stream(stream_id: int):
    """Stop a stream."""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    stream.stop()
    return {"success": True, "message": f"Stream {stream_id} stopped"}


@app.delete("/api/v1/streams/{stream_id}")
async def delete_stream(stream_id: int):
    """Delete a stream."""
    if stream_id not in stream_manager.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    stream_manager.remove_stream(stream_id)
    return {"success": True, "message": f"Stream {stream_id} deleted"}


@app.get("/api/v1/streams/{stream_id}/frame")
async def get_frame(stream_id: int):
    """Get latest frame as JPEG image."""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    jpeg = stream.get_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="No frame available")
    
    return StreamingResponse(
        iter([jpeg]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/api/v1/streams/{stream_id}/mjpeg")
async def mjpeg_stream(stream_id: int):
    """Get MJPEG video stream."""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    async def generate():
        while True:
            jpeg = stream.get_jpeg()
            if jpeg:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                )
            await asyncio.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/v1/streams/{stream_id}/prediction")
async def get_prediction(stream_id: int):
    """Get latest violence prediction for a stream."""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    if not stream.last_prediction:
        return {"success": True, "data": None, "message": "No prediction yet"}
    
    return {"success": True, "data": stream.last_prediction}


@app.get("/api/v1/inference/scores")
async def get_all_scores():
    """Get latest inference scores for all streams."""
    scores = []
    for stream in stream_manager.streams.values():
        if stream.last_prediction:
            scores.append(stream.last_prediction)
    return {"success": True, "data": scores}


@app.get("/api/v1/model/status")
async def get_model_status():
    """Get ML model status."""
    return {
        "success": True,
        "data": {
            "is_loaded": detector.is_loaded,
            "model_path": str(MODEL_PATH),
            "threshold": VIOLENCE_THRESHOLD,
            "alert_threshold": VIOLENCE_ALERT_THRESHOLD,
            "alert_cooldown": VIOLENCE_ALERT_COOLDOWN,
            "inference_interval": INFERENCE_INTERVAL
        }
    }


# ============== WebSocket for real-time predictions ==============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time violence predictions."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total: {len(active_connections)}")
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(active_connections)}")


# ============== Run ==============

if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )

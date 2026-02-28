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
import json
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from uuid import uuid4
from enum import Enum

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from loguru import logger
from sqlalchemy import select, update

# Import face extractor
from app.detection.face_extractor import get_face_extractor

# Import database modules for persistence
from app.database import (
    init_db, async_session, 
    Event, Stream as DBStream, EventStatus, AlertSeverity
)
from app.config import settings


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


# ============== Model Configuration ==============
MODEL_PATH = Path(__file__).parent / ".." / "ml-service" / "models" / "violence_model_legacy.h5"
EXPECTED_FRAMES = 16
TARGET_SIZE = (224, 224)
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "0.3"))  # Run inference every 0.3 seconds (faster detection)
VIOLENCE_THRESHOLD = float(os.getenv("VIOLENCE_THRESHOLD", "0.50"))  # 50% for is_violent flag
VIOLENCE_ALERT_THRESHOLD = float(os.getenv("VIOLENCE_ALERT_THRESHOLD", "0.80"))  # Alert at 90%+ (instant notification)
VIOLENCE_ALERT_COOLDOWN = float(os.getenv("VIOLENCE_ALERT_COOLDOWN", "5.0"))  # 5 second cooldown between alerts

# Model prediction smoothing (reduce false positives)
PREDICTION_SMOOTHING_WINDOW = int(os.getenv("PREDICTION_SMOOTHING_WINDOW", "3"))  # Average last N predictions
CONSECUTIVE_DETECTIONS_REQUIRED = int(os.getenv("CONSECUTIVE_DETECTIONS_REQUIRED", "2"))  # N consecutive high scores to trigger alert

# Clip recording settings
CLIP_BUFFER_SECONDS = int(os.getenv("CLIP_BUFFER_SECONDS", "10"))  # 10s before violence
CLIP_AFTER_SECONDS = int(os.getenv("CLIP_AFTER_SECONDS", "10"))  # 10s after violence ends
CLIP_MAX_DURATION = int(os.getenv("CLIP_MAX_DURATION", "60"))  # Max 60s violence duration before force-save
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", "./clips"))
CLIPS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR = CLIPS_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(exist_ok=True)
STREAM_FPS = 30  # Default FPS for clip recording (overridden by actual measured FPS)


# ============== Helper Functions ==============

def verify_clip_file(file_path: Optional[str]) -> Optional[str]:
    """
    Return the file path only if the file exists, otherwise return None.
    This ensures we don't return references to deleted files.
    """
    if not file_path:
        return None
    
    # Check in clips directory
    full_path = CLIPS_DIR / file_path
    if full_path.exists():
        return file_path
    
    # Check if it's already an absolute path or in thumbnails dir
    if Path(file_path).exists():
        return file_path
    
    # Check in thumbnails subdirectory
    thumb_path = THUMBNAILS_DIR / file_path
    if thumb_path.exists():
        return file_path
    
    return None


# ============== Violence Detection Model ==============

class ViolenceDetector:
    """Loads and runs the violence detection model."""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self._lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model."""
        try:
            model_path = MODEL_PATH.resolve()
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}")
                return
            
            import tensorflow as tf
            from tensorflow import keras
            
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            
            # Try direct load first
            try:
                self.model = keras.models.load_model(str(model_path), compile=False)
                self.is_loaded = True
                logger.info(f"✅ Loaded violence detection model from {model_path}")
                return
            except Exception as e:
                logger.warning(f"Direct load failed: {str(e)[:80]}, trying fallback...")
            
            # Fallback: Build architecture and load weights
            from tensorflow.keras import layers
            
            input_shape = (EXPECTED_FRAMES, *TARGET_SIZE, 3)
            inputs = keras.Input(shape=input_shape)
            
            base_model = keras.applications.MobileNetV2(
                weights=None, include_top=False, input_shape=(224, 224, 3)
            )
            
            x = layers.TimeDistributed(base_model)(inputs)
            x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
            x = layers.LSTM(64)(x)
            x = layers.Dense(64, activation='relu')(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            self.model = keras.Model(inputs=inputs, outputs=outputs)
            self.model.load_weights(str(model_path))
            self.is_loaded = True
            logger.info(f"✅ Loaded model weights from {model_path}")
            
            # Warmup
            dummy = np.zeros((1, EXPECTED_FRAMES, *TARGET_SIZE, 3), dtype=np.float32)
            self.model.predict(dummy, verbose=0)
            logger.info("✅ Model warmup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
    
    def predict(self, frames: List[np.ndarray]) -> Optional[dict]:
        """Run prediction on frames."""
        if not self.is_loaded or self.model is None:
            return None
        
        with self._lock:
            try:
                # Preprocess frames
                processed = self._preprocess(frames)
                
                # Run inference
                start = time.time()
                prediction = self.model.predict(processed, verbose=0)
                inference_time = (time.time() - start) * 1000
                
                # Parse result
                if prediction.shape[-1] == 2:
                    violence_score = float(prediction[0][0])
                else:
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
                return None
    
    def _preprocess(self, frames: List[np.ndarray]) -> np.ndarray:
        """Preprocess frames for model input."""
        # Ensure we have exactly EXPECTED_FRAMES
        if len(frames) < EXPECTED_FRAMES:
            frames = list(frames) + [frames[-1]] * (EXPECTED_FRAMES - len(frames))
        elif len(frames) > EXPECTED_FRAMES:
            indices = np.linspace(0, len(frames) - 1, EXPECTED_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]
        
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            processed.append(normalized)
        
        stacked = np.stack(processed, axis=0)
        return np.expand_dims(stacked, axis=0)


# Global detector instance
detector = ViolenceDetector()


# ============== Event State & Clip Recording ==============

class EventPhase(Enum):
    """Violence event detection phases."""
    IDLE = "idle"           # No violence detected
    VIOLENCE = "violence"   # Violence in progress
    POST_BUFFER = "post"    # Recording post-violence buffer


@dataclass
class ViolenceEventState:
    """Tracks state of a violence event for clip recording."""
    event_id: str
    stream_id: int
    stream_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    max_score: float = 0.0
    frame_scores: List[float] = field(default_factory=list)
    pre_buffer_frames: List[Tuple[np.ndarray, float]] = field(default_factory=list)  # (frame, timestamp)
    event_frames: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    post_buffer_frames: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    face_paths: List[str] = field(default_factory=list)  # Detected participant faces


class EventRecorder:
    """Records violence event clips with pre/post buffers."""
    
    def __init__(self, stream_id: int, stream_name: str):
        self.stream_id = stream_id
        self.stream_name = stream_name
        self.phase = EventPhase.IDLE
        self.current_event: Optional[ViolenceEventState] = None
        
        # Rolling buffer for 10s before violence (frames with timestamps)
        buffer_size = CLIP_BUFFER_SECONDS * STREAM_FPS
        self.pre_buffer: deque = deque(maxlen=buffer_size)
        
        # Post-buffer countdown
        self.post_buffer_start: Optional[float] = None
        
        # Violence start time for max duration check
        self.violence_start_time: Optional[float] = None
        self._lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to rolling pre-buffer and collect frames during violence/post-buffer phases."""
        with self._lock:
            # Always add to pre-buffer (rolling window for 10s before violence)
            self.pre_buffer.append((frame.copy(), timestamp))
            
            # If in violence phase, collect all frames for the clip
            if self.phase == EventPhase.VIOLENCE and self.current_event:
                self.current_event.event_frames.append((frame.copy(), timestamp))
                
                # Check if max violence duration exceeded - force end event
                if self.violence_start_time:
                    violence_duration = timestamp - self.violence_start_time
                    if violence_duration >= CLIP_MAX_DURATION:
                        logger.warning(f"⚠️ Max violence duration ({CLIP_MAX_DURATION}s) exceeded, force-saving clip")
                        self._end_violence(timestamp)
            
            # If in post-buffer phase, collect frames
            elif self.phase == EventPhase.POST_BUFFER and self.current_event:
                self.current_event.post_buffer_frames.append((frame.copy(), timestamp))
                
                # Check if post-buffer time elapsed
                if self.post_buffer_start is not None:
                    elapsed = timestamp - self.post_buffer_start
                    if elapsed >= CLIP_AFTER_SECONDS:
                        self._finalize_event()
    
    def on_prediction(self, score: float, frame: np.ndarray, timestamp: float):
        """Process prediction result."""
        with self._lock:
            is_violent = score >= VIOLENCE_ALERT_THRESHOLD
            
            if self.phase == EventPhase.IDLE:
                if is_violent:
                    # Violence started - begin event
                    self._start_event(score, timestamp)
                    
            elif self.phase == EventPhase.VIOLENCE:
                if is_violent:
                    # Violence continues - update scores (frames are collected by add_frame())
                    if self.current_event is not None:
                        self.current_event.frame_scores.append(score)
                        self.current_event.max_score = max(self.current_event.max_score, score)
                else:
                    # Violence ended - start post-buffer
                    self._end_violence(timestamp)
                    
            elif self.phase == EventPhase.POST_BUFFER:
                if is_violent:
                    # Violence resumed - go back to violence phase
                    if self.current_event is not None:
                        self.phase = EventPhase.VIOLENCE
                        self.current_event.end_time = None
                        self.post_buffer_start = None
                        # Move post_buffer_frames to event_frames
                        self.current_event.event_frames.extend(self.current_event.post_buffer_frames)
                        self.current_event.post_buffer_frames = []
                        self.current_event.frame_scores.append(score)
                        self.current_event.max_score = max(self.current_event.max_score, score)
    
    def _start_event(self, score: float, timestamp: float):
        """Start a new violence event."""
        event_id = str(uuid4())
        self.current_event = ViolenceEventState(
            event_id=event_id,
            stream_id=self.stream_id,
            stream_name=self.stream_name,
            start_time=datetime.utcnow(),
            max_score=score,
            frame_scores=[score],
            pre_buffer_frames=list(self.pre_buffer),
        )
        self.phase = EventPhase.VIOLENCE
        self.violence_start_time = timestamp
        logger.info(f"🔴 Violence event started: {event_id} on {self.stream_name} (score: {score:.0%}, pre-buffer: {len(self.pre_buffer)} frames)")
        
        # Broadcast event_start
        broadcast_event_start(self.current_event)
    
    def _end_violence(self, timestamp: float):
        """Violence ended, start post-buffer recording."""
        if self.current_event:
            self.current_event.end_time = datetime.utcnow()
            self.phase = EventPhase.POST_BUFFER
            self.post_buffer_start = timestamp
            self.violence_start_time = None
            event_frames_count = len(self.current_event.event_frames)
            logger.info(f"🟡 Violence ended, collected {event_frames_count} event frames, recording {CLIP_AFTER_SECONDS}s post-buffer...")
    
    def _finalize_event(self):
        """Finalize event and save clip."""
        if not self.current_event:
            return
        
        event = self.current_event
        total_frames = len(event.pre_buffer_frames) + len(event.event_frames) + len(event.post_buffer_frames)
        logger.info(f"🎬 Finalizing event {event.event_id}: {len(event.pre_buffer_frames)} pre + {len(event.event_frames)} event + {len(event.post_buffer_frames)} post = {total_frames} total frames")
        
        # Save clip in background thread
        threading.Thread(
            target=self._save_clip,
            args=(event,),
            daemon=True
        ).start()
        
        # Reset state
        self.phase = EventPhase.IDLE
        self.current_event = None
        self.post_buffer_start = None
        self.violence_start_time = None
    
    def _save_clip(self, event: ViolenceEventState):
        """Save video clip from collected frames with correct real-time playback speed."""
        try:
            # Combine all frames: pre-buffer + event + post-buffer
            all_frames = []
            all_frames.extend(event.pre_buffer_frames)
            all_frames.extend(event.event_frames)
            all_frames.extend(event.post_buffer_frames)
            
            logger.info(f"📼 Saving clip for event {event.event_id}: "
                       f"{len(event.pre_buffer_frames)} pre + {len(event.event_frames)} event + "
                       f"{len(event.post_buffer_frames)} post = {len(all_frames)} total frames")
            
            if not all_frames:
                logger.warning(f"❌ No frames to save for event {event.event_id}")
                return
            
            # Calculate actual FPS from frame timestamps for real-time playback
            # This prevents the fast-forwarded effect caused by using a hardcoded FPS
            actual_fps = STREAM_FPS  # default fallback
            if len(all_frames) >= 2:
                first_ts = all_frames[0][1]
                last_ts = all_frames[-1][1]
                elapsed = last_ts - first_ts
                if elapsed > 0:
                    actual_fps = len(all_frames) / elapsed
                    # Clamp to reasonable range (5-60 fps)
                    actual_fps = max(5.0, min(60.0, actual_fps))
                    logger.info(f"📊 Measured actual FPS: {actual_fps:.1f} (from {len(all_frames)} frames over {elapsed:.1f}s)")
            
            # Get frame dimensions from first frame
            first_frame = all_frames[0][0]
            height, width = first_frame.shape[:2]
            logger.debug(f"Frame dimensions: {width}x{height}")
            
            # Generate filename
            timestamp_str = event.start_time.strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c if c.isalnum() else "_" for c in event.stream_name)
            clip_filename = f"{timestamp_str}_{safe_name}_{event.event_id[:8]}.mp4"
            clip_path = CLIPS_DIR / clip_filename
            
            # Save thumbnail (middle frame of event, or first event frame if no middle)
            thumb_filename = f"{timestamp_str}_{safe_name}_{event.event_id[:8]}.jpg"
            thumb_path = THUMBNAILS_DIR / thumb_filename
            
            event_start_idx = len(event.pre_buffer_frames)
            event_end_idx = event_start_idx + len(event.event_frames)
            mid_idx = (event_start_idx + event_end_idx) // 2
            
            # Ensure we have a valid index for thumbnail
            if mid_idx >= len(all_frames):
                mid_idx = len(all_frames) // 2
            if mid_idx < len(all_frames):
                success = cv2.imwrite(str(thumb_path), all_frames[mid_idx][0])
                if success:
                    # Store relative path from clips dir (including thumbnails/ subdirectory)
                    event.thumbnail_path = f"thumbnails/{thumb_filename}"
                    logger.info(f"📸 Saved thumbnail: {thumb_filename}")
                else:
                    logger.error(f"❌ Failed to save thumbnail: {thumb_path}")
            
            # Use the measured actual FPS for the video writer so playback is real-time
            write_fps = int(round(actual_fps))
            
            # Write video using H.264 codec for browser compatibility
            fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(clip_path), fourcc_h264, write_fps, (width, height))
            
            if not out.isOpened():
                # Fallback: try other H.264 fourcc codes
                for codec in ['H264', 'x264', 'X264']:
                    fourcc_alt = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(str(clip_path), fourcc_alt, write_fps, (width, height))
                    if out.isOpened():
                        logger.info(f"Using fallback H.264 codec: {codec}")
                        break
            
            if not out.isOpened():
                # Last resort: use mp4v (won't play in browsers but at least saves)
                logger.warning("⚠️ H.264 codecs unavailable, falling back to mp4v (clips won't play in browser)")
                fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(clip_path), fourcc_mp4v, write_fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"❌ Failed to open VideoWriter for {clip_path}")
                return
            
            frames_written = 0
            for frame, _ in all_frames:
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            # Verify file was created
            if not clip_path.exists():
                logger.error(f"❌ Clip file not created: {clip_path}")
                return
            
            file_size = clip_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is likely corrupt
                logger.error(f"❌ Clip file too small ({file_size} bytes): {clip_path}")
                return
            
            # Calculate real duration from timestamps
            clip_duration = len(all_frames) / actual_fps
            event.clip_path = clip_filename
            
            logger.info(f"✅ Saved clip: {clip_filename} ({clip_duration:.1f}s, {frames_written} frames, {file_size/1024:.1f}KB)")
            
            # Extract faces from the saved clip
            try:
                face_extractor = get_face_extractor()
                event.face_paths = face_extractor.process_clip(str(clip_path), event.event_id)
                logger.info(f"👤 Extracted {len(event.face_paths)} participant faces")
            except Exception as fe:
                logger.warning(f"Face extraction failed: {fe}")
                event.face_paths = []
            
            # Broadcast event completion with clip info
            broadcast_event_end(event, clip_duration)
            
        except Exception as e:
            logger.error(f"❌ Failed to save clip for event {event.event_id}: {e}")
            import traceback
            traceback.print_exc()


def broadcast_event_start(event: ViolenceEventState):
    """Broadcast event_start to WebSocket clients."""
    alert = {
        "type": "event_start",
        "event_id": event.event_id,
        "stream_id": str(event.stream_id),
        "stream_name": event.stream_name,
        "start_time": event.start_time.isoformat(),
        "timestamp": datetime.utcnow().isoformat(),
        "confidence": event.max_score,
        "max_score": event.max_score,
        "severity": "critical" if event.max_score >= 0.90 else "high",
        "status": "PENDING",
        "message": f"Violence detected on {event.stream_name} ({event.max_score * 100:.0f}% confidence)",
    }
    _broadcast_ws("event_start", alert)


def broadcast_event_end(event: ViolenceEventState, clip_duration: float):
    """Broadcast event_end with clip info to WebSocket clients."""
    avg_score = sum(event.frame_scores) / len(event.frame_scores) if event.frame_scores else 0
    
    alert = {
        "type": "violence_alert",
        "event_id": event.event_id,
        "stream_id": str(event.stream_id),
        "stream_name": event.stream_name,
        "start_time": event.start_time.isoformat(),
        "end_time": event.end_time.isoformat() if event.end_time else None,
        "timestamp": datetime.utcnow().isoformat(),
        "confidence": event.max_score,
        "max_score": event.max_score,
        "max_confidence": event.max_score,
        "avg_confidence": avg_score,
        "avg_score": avg_score,
        "severity": "critical" if event.max_score >= 0.90 else "high",
        "status": "PENDING",  # Initial status for review
        "message": f"Violence detected on {event.stream_name} ({event.max_score * 100:.0f}% confidence)",
        "clip_path": event.clip_path,
        "thumbnail_path": event.thumbnail_path,
        "clip_duration": clip_duration,
        "duration": (event.end_time - event.start_time).total_seconds() if event.end_time else 0,
        "duration_seconds": (event.end_time - event.start_time).total_seconds() if event.end_time else 0,
        "face_paths": event.face_paths,  # Detected participant faces
        "participants_count": len(event.face_paths),
    }
    _broadcast_ws("violence_alert", alert)
    
    # Store event in database for persistence (using run_coroutine_threadsafe since we're in a background thread)
    global main_event_loop
    if main_event_loop:
        asyncio.run_coroutine_threadsafe(store_event_async(alert), main_event_loop)
    else:
        logger.warning("No main event loop available - falling back to sync storage")
        store_event(alert)


# In-memory event storage (kept for backward compatibility, but DB is primary)
stored_events: List[dict] = []
MAX_STORED_EVENTS = 100


async def store_event_async(event: dict):
    """Store event in PostgreSQL database."""
    try:
        async with async_session() as session:
            # Get confidence - try multiple field names for compatibility
            confidence = event.get("max_confidence") or event.get("max_score") or event.get("peak_confidence") or event.get("confidence", 0)
            avg_confidence = event.get("avg_confidence") or event.get("avg_score") or confidence
            
            # Determine severity based on confidence
            if confidence >= 0.95:
                severity = AlertSeverity.CRITICAL
            elif confidence >= 0.85:
                severity = AlertSeverity.HIGH
            elif confidence >= 0.7:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            # Parse datetime from ISO string
            start_time = datetime.fromisoformat(event.get("start_time", datetime.utcnow().isoformat()))
            end_time = datetime.fromisoformat(event.get("end_time", datetime.utcnow().isoformat())) if event.get("end_time") else None
            
            # Convert stream_id to int (may be passed as string)
            stream_id_raw = event.get("stream_id", 0)
            stream_id = int(stream_id_raw) if stream_id_raw else 0
            
            db_event = Event(
                stream_id=stream_id,
                stream_name=event.get("stream_name", "Unknown"),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=event.get("duration_seconds") or event.get("duration"),
                max_confidence=confidence,
                avg_confidence=avg_confidence,
                min_confidence=confidence,
                frame_count=1,
                severity=severity,
                status=EventStatus.PENDING,
                clip_path=event.get("clip_path"),
                clip_duration=event.get("clip_duration"),
                thumbnail_path=event.get("thumbnail_path"),
            )
            session.add(db_event)
            await session.commit()
            logger.info(f"📝 Event stored in database: stream={event.get('stream_name')}, confidence={confidence:.1%}, clip={event.get('clip_path')}")
    except Exception as e:
        logger.error(f"Failed to store event in database: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to in-memory storage
        stored_events.insert(0, event)
        stored_events[:] = stored_events[:MAX_STORED_EVENTS]


def store_event(event: dict):
    """Store event - wrapper for sync calls (deprecated, use store_event_async)."""
    global stored_events
    stored_events.insert(0, event)
    stored_events = stored_events[:MAX_STORED_EVENTS]


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
        self.frame_buffer: deque = deque(maxlen=60)  # ~2 seconsds at 30fps
        
        # Latest prediction
        self.last_prediction: Optional[dict] = None
        self.prediction_callback: Optional[Callable[[dict], None]] = None  # Set by manager
        
        # Cached JPEG for real-time streaming (encoded in capture loop)
        self._last_jpeg: Optional[bytes] = None
        self._last_jpeg_with_overlay: Optional[bytes] = None
        self._jpeg_encode_quality = 50  # Minimum quality for fastest encoding (real-time)
        self._jpeg_encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
        
        # Violence alert cooldown tracking
        self._last_violence_alert_time = 0.0
        
        # Prediction smoothing to reduce false positives
        self._recent_scores: deque = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
        self._consecutive_high_count = 0
        
        # Event recorder for clip generation
        self.event_recorder = EventRecorder(stream_id, name)
    
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
        
        # Finalize any pending violence event before stopping
        if self.event_recorder.phase != EventPhase.IDLE and self.event_recorder.current_event:
            logger.info(f"🛑 Stream stopping, finalizing pending violence event...")
            self.event_recorder._end_violence(time.time())
            # Force immediate finalization
            self.event_recorder._finalize_event()
        
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
        """Main capture loop - REAL-TIME with zero delay."""
        while self.is_running:
            try:
                if self.capture is None or not self.capture.isOpened():
                    self._connect()
                    continue
                
                # Flush any buffered frames to get the latest
                # This is critical for real-time - discard old frames
                self.capture.grab()  # Grab without decode to flush buffer
                
                # Now read the actual latest frame
                ret, frame = self.capture.read()
                if ret:
                    current_time = time.time()
                    
                    # Fast JPEG encode with minimal quality for real-time
                    _, jpeg = cv2.imencode('.jpg', frame, self._jpeg_encode_params)
                    jpeg_bytes = jpeg.tobytes()
                    
                    # Create overlay directly on frame (no copy for speed)
                    score = self.last_prediction.get('violence_score', 0) if self.last_prediction else 0
                    score_pct = int(score * 100)
                    
                    # Draw minimal overlay for speed
                    color = (0, 0, 255) if score > 0.65 else (0, 255, 255) if score > 0.4 else (0, 255, 0)
                    cv2.rectangle(frame, (10, 10), (160, 45), (0, 0, 0), -1)
                    cv2.putText(frame, f"{score_pct}%", (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    
                    _, overlay_jpeg = cv2.imencode('.jpg', frame, self._jpeg_encode_params)
                    overlay_jpeg_bytes = overlay_jpeg.tobytes()
                    
                    # Minimal lock time - just swap pointers
                    with self._lock:
                        self.last_frame = frame
                        self._last_jpeg = jpeg_bytes
                        self._last_jpeg_with_overlay = overlay_jpeg_bytes
                        self.frame_buffer.append(frame)
                        self.frame_count += 1
                        self.is_connected = True
                        self.error = None
                    
                    # Feed frame to event recorder
                    self.event_recorder.add_frame(frame, current_time)
                else:
                    self.is_connected = False
                    self._connect()
                    
            except Exception as e:
                self.error = str(e)
                self.is_connected = False
                time.sleep(0.5)
    
    def _inference_loop(self):
        """Run inference periodically on buffered frames with smoothing."""
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
                    raw_score = result["violence_score"]
                    
                    # Apply temporal smoothing to reduce false positives
                    self._recent_scores.append(raw_score)
                    smoothed_score = sum(self._recent_scores) / len(self._recent_scores)
                    
                    # Update result with smoothed score
                    result["raw_score"] = raw_score
                    result["violence_score"] = smoothed_score
                    result["non_violence_score"] = 1.0 - smoothed_score
                    result["is_violent"] = smoothed_score >= VIOLENCE_THRESHOLD
                    result["stream_id"] = str(self.id)
                    result["stream_name"] = self.name
                    
                    # Track consecutive high scores
                    if raw_score >= VIOLENCE_ALERT_THRESHOLD:
                        self._consecutive_high_count += 1
                    else:
                        self._consecutive_high_count = 0
                    
                    self.last_prediction = result
                    
                    # Simplified logging - only log significant events
                    if smoothed_score >= VIOLENCE_THRESHOLD:
                        logger.warning(f"[{self.name}] Violence: {smoothed_score:.0%}")
                    
                    # Trigger callback for WebSocket broadcast
                    if self.prediction_callback:
                        self.prediction_callback(result)
                    
                    # Feed RAW score to event recorder (use raw score for detection to catch spikes)
                    # This ensures we don't miss violence events due to smoothing
                    current_frame = frames[-1].copy() if frames else None
                    if current_frame is not None:
                        self.event_recorder.on_prediction(raw_score, current_frame, time.time())
                    
                    # NOTE: Per-frame violence_alert emission removed.
                    # The EventRecorder handles the alert lifecycle properly:
                    #   - broadcast_event_start() when violence begins (one alert)
                    #   - broadcast_event_end() when clip is saved (one alert)
                    # _maybe_emit_violence_alert was causing repeated alerts every 5s.
                        
            except Exception as e:
                logger.error(f"Inference error: {e}")
    
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
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": score,
            "severity": severity,
            "message": f"Violence detected on {self.name} ({score * 100:.0f}%)",
        }
        
        logger.warning(f"ALERT: {self.name} - {score:.0%}")
        broadcast_violence_alert(alert)
    
    def _connect(self):
        """Connect to the RTSP stream with low latency settings."""
        try:
            if self.capture:
                self.capture.release()
            
            logger.info(f"Connecting to: {self.url}")
            
            # Use FFmpeg backend with aggressive low-latency options
            # REAL-TIME FFmpeg options - UDP for lowest latency
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;udp|'  # UDP = lowest latency (no TCP handshake)
                'fflags;nobuffer+discardcorrupt+fastseek|'  # No buffering at all
                'flags;low_delay|'  # Low delay decoding
                'framedrop;1|'  # Drop frames if behind
                'max_delay;0|'  # Zero delay
                'reorder_queue_size;0|'  # No packet reordering
                'analyzeduration;100000|'  # 0.1s analyze (minimal)
                'probesize;32768|'  # Tiny probe (32KB)
                'flush_packets;1|'  # Flush immediately
                'avioflags;direct|'  # Direct I/O
                'timeout;2000000'  # 2s timeout
            )
            
            self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # ZERO buffer for real-time
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            
            # Native resolution for speed (no resize overhead)
            # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Match camera FPS
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Fast connection - wait only 3 seconds
            connect_start = time.time()
            while time.time() - connect_start < 3:
                if self.capture.isOpened():
                    # Try to read a test frame to confirm connection
                    ret, _ = self.capture.read()
                    if ret:
                        self.is_connected = True
                        self.error = None
                        logger.info(f"✓ Connected: {self.name}")
                        return
                time.sleep(0.1)  # Check more frequently
            
            # Connection timed out
            self.is_connected = False
            self.error = "Connection timed out - check RTSP URL"
            logger.warning(f"Connection timeout: {self.name} - {self.url}")
            time.sleep(1)  # Shorter retry delay
                
        except Exception as e:
            self.error = str(e)
            self.is_connected = False
            time.sleep(1)  # Shorter retry delay
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        with self._lock:
            return self.last_frame.copy() if self.last_frame is not None else None
    
    def get_jpeg(self, with_overlay: bool = False) -> Optional[bytes]:
        """Get the latest frame as JPEG bytes (pre-encoded for low latency)."""
        with self._lock:
            if with_overlay:
                return self._last_jpeg_with_overlay
            return self._last_jpeg
    
    def get_status(self) -> dict:
        """Get stream status in frontend-compatible format."""
        # Determine status string
        if self.error:
            status = "error"
        elif self.is_running and self.is_connected:
            status = "running"
        elif self.is_running and not self.is_connected:
            status = "connecting"
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

async def load_streams_from_db():
    """Load active streams from database."""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(DBStream).where(DBStream.is_active == True)
            )
            db_streams = result.scalars().all()
            
            for db_stream in db_streams:
                logger.info(f"Loaded stream from DB: {db_stream.name} (ID: {db_stream.id})")
                # Create stream in manager (but don't auto-start)
                stream = SimpleRTSPStream(
                    stream_id=db_stream.id,
                    name=db_stream.name,
                    url=db_stream.url,
                )
                stream_manager.streams[db_stream.id] = stream
                
            logger.info(f"✅ Loaded {len(db_streams)} streams from database")
    except Exception as e:
        logger.error(f"Failed to load streams from DB: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    logger.info("🚀 Starting RTSP Service with Database Persistence...")
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Load streams from database
    await load_streams_from_db()
    
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
    """Add a new RTSP stream and persist to database."""
    # Save to database first
    try:
        async with async_session() as session:
            db_stream = DBStream(
                name=request.name,
                url=request.url,
                stream_type="rtsp",
                is_active=True,
            )
            session.add(db_stream)
            await session.commit()
            await session.refresh(db_stream)
            stream_id = db_stream.id
            logger.info(f"📝 Stream saved to database: {request.name} (ID: {stream_id})")
    except Exception as e:
        logger.error(f"Failed to save stream to database: {e}")
        # Fallback to memory-only
        stream_id = stream_manager.add_stream(
            name=request.name,
            url=request.url,
            auto_start=request.auto_start
        )
        return {"success": True, "stream_id": str(stream_id)}
    
    # Create stream in manager
    stream = SimpleRTSPStream(
        stream_id=stream_id,
        name=request.name,
        url=request.url,
    )
    stream_manager.streams[stream_id] = stream
    
    if request.auto_start:
        stream.start()
    
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
    """Delete a stream from database and memory."""
    if stream_id not in stream_manager.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Remove from database
    try:
        async with async_session() as session:
            await session.execute(
                update(DBStream)
                .where(DBStream.id == stream_id)
                .values(is_active=False)
            )
            await session.commit()
            logger.info(f"📝 Stream {stream_id} marked inactive in database")
    except Exception as e:
        logger.error(f"Failed to update stream in database: {e}")
    
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
async def mjpeg_stream(stream_id: int, overlay: bool = True):
    """Get real-time MJPEG video stream with optional violence score overlay.
    
    Args:
        stream_id: Stream ID
        overlay: If True, includes violence score overlay on video (default: True)
    """
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    async def generate():
        """Zero-delay frame generator - push every new frame immediately."""
        last_id = 0
        while True:
            jpeg = stream.get_jpeg(with_overlay=overlay)
            frame_id = stream.frame_count
            if jpeg and frame_id != last_id:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n'
                )
                last_id = frame_id
            await asyncio.sleep(0)  # Yield immediately, no delay
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no"
        }
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


# ============== Event & Clip API Routes ==============

@app.get("/api/v1/events")
async def get_events(limit: int = 50, offset: int = 0, status: Optional[str] = None):
    """Get violence events from database."""
    try:
        async with async_session() as session:
            query = select(Event).order_by(Event.created_at.desc())
            
            if status:
                try:
                    event_status = EventStatus(status)
                    query = query.where(Event.status == event_status)
                except ValueError:
                    pass  # Invalid status, ignore filter
            
            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            events = result.scalars().all()
            
            # Get total count for pagination
            count_query = select(Event)
            if status:
                try:
                    event_status = EventStatus(status)
                    count_query = count_query.where(Event.status == event_status)
                except ValueError:
                    pass
            count_result = await session.execute(count_query)
            total_count = len(count_result.scalars().all())
            
            # Build event list, checking if clip files exist
            event_list = []
            for e in events:
                clip_exists = verify_clip_file(e.clip_path)
                thumb_exists = verify_clip_file(e.thumbnail_path)
                event_list.append({
                    "id": e.id,
                    "event_id": str(e.id),  # For compatibility
                    "stream_id": e.stream_id,
                    "stream_name": e.stream_name,
                    "start_time": e.start_time.isoformat() if e.start_time else None,
                    "end_time": e.end_time.isoformat() if e.end_time else None,
                    "duration_seconds": e.duration_seconds if clip_exists else None,
                    "max_confidence": e.max_confidence,
                    "peak_confidence": e.max_confidence,  # Alias
                    "avg_confidence": e.avg_confidence,
                    "severity": e.severity.value if e.severity else None,
                    "status": e.status.value if e.status else None,
                    "clip_path": clip_exists,
                    "clip_duration": e.clip_duration if clip_exists else None,
                    "thumbnail_path": thumb_exists,
                    "created_at": e.created_at.isoformat() if e.created_at else None
                })
            
            return {
                "success": True,
                "data": event_list,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "count": total_count
                }
            }
    except Exception as e:
        logger.error(f"Failed to get events from database: {e}")
        # Fallback to in-memory events
        events = stored_events[offset:offset + limit]
        return {
            "success": True,
            "data": events,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "count": len(stored_events)
            }
        }


@app.get("/api/v1/events/{event_id}")
async def get_event(event_id: str):
    """Get a specific violence event from database."""
    try:
        async with async_session() as session:
            event_id_int = int(event_id)
            result = await session.execute(
                select(Event).where(Event.id == event_id_int)
            )
            e = result.scalar_one_or_none()
            if e:
                clip_exists = verify_clip_file(e.clip_path)
                thumb_exists = verify_clip_file(e.thumbnail_path)
                return {
                    "success": True,
                    "data": {
                        "id": e.id,
                        "event_id": str(e.id),
                        "stream_id": e.stream_id,
                        "stream_name": e.stream_name,
                        "start_time": e.start_time.isoformat() if e.start_time else None,
                        "end_time": e.end_time.isoformat() if e.end_time else None,
                        "duration_seconds": e.duration_seconds if clip_exists else None,
                        "max_confidence": e.max_confidence,
                        "avg_confidence": e.avg_confidence,
                        "severity": e.severity.value if e.severity else None,
                        "status": e.status.value if e.status else None,
                        "clip_path": clip_exists,
                        "clip_duration": e.clip_duration if clip_exists else None,
                        "thumbnail_path": thumb_exists,
                        "created_at": e.created_at.isoformat() if e.created_at else None
                    }
                }
    except (ValueError, Exception) as e:
        logger.error(f"Failed to get event from database: {e}")
    
    # Fallback to in-memory
    for event in stored_events:
        if event.get("event_id") == event_id:
            return {"success": True, "data": event}
    raise HTTPException(status_code=404, detail="Event not found")


@app.post("/api/v1/events/{event_id}/action-executed")
async def mark_action_executed(event_id: str):
    """Mark event as action executed in database."""
    try:
        async with async_session() as session:
            event_id_int = int(event_id)
            await session.execute(
                update(Event)
                .where(Event.id == event_id_int)
                .values(
                    status=EventStatus.ACTION_EXECUTED,
                    reviewed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
            )
            await session.commit()
            return {"success": True, "message": "Event marked as action executed"}
    except (ValueError, Exception) as e:
        logger.error(f"Failed to update event in database: {e}")
    
    # Fallback to in-memory
    for event in stored_events:
        if event.get("event_id") == event_id:
            event["status"] = "ACTION_EXECUTED"
            event["reviewed_at"] = datetime.utcnow().isoformat()
            return {"success": True, "message": "Event marked as action executed"}
    raise HTTPException(status_code=404, detail="Event not found")


@app.post("/api/v1/events/{event_id}/no-action-required")
async def mark_no_action_required(event_id: str):
    """Mark event as no action required in database."""
    try:
        async with async_session() as session:
            event_id_int = int(event_id)
            await session.execute(
                update(Event)
                .where(Event.id == event_id_int)
                .values(
                    status=EventStatus.NO_ACTION_REQUIRED,
                    reviewed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
            )
            await session.commit()
            return {"success": True, "message": "Event marked as no action required"}
    except (ValueError, Exception) as e:
        logger.error(f"Failed to update event in database: {e}")
    
    # Fallback to in-memory
    for event in stored_events:
        if event.get("event_id") == event_id:
            event["status"] = "NO_ACTION_REQUIRED"
            event["reviewed_at"] = datetime.utcnow().isoformat()
            return {"success": True, "message": "Event marked as no action required"}
    raise HTTPException(status_code=404, detail="Event not found")


@app.post("/api/v1/events/import-clips")
async def import_clips_as_events():
    """Import existing clip files as events in the database.
    
    This is useful for recovering events that were lost due to bugs or restarts.
    Parses clip filenames to extract stream name and timestamp.
    """
    try:
        import re
        imported = 0
        skipped = 0
        errors = []
        
        async with async_session() as session:
            # Get existing clip paths to avoid duplicates
            result = await session.execute(select(Event.clip_path))
            existing_clips = {row[0] for row in result.fetchall() if row[0]}
            
            # Scan clips directory
            for clip_file in CLIPS_DIR.glob("*.mp4"):
                clip_filename = clip_file.name
                
                # Skip if already in database
                if clip_filename in existing_clips:
                    skipped += 1
                    continue
                
                try:
                    # Parse filename: YYYYMMDD_HHMMSS_StreamName_EventId.mp4
                    # or: StreamName_EventId_YYYYMMDD_HHMMSS_type.mp4
                    name = clip_file.stem
                    
                    # Try format: YYYYMMDD_HHMMSS_StreamName_UUID
                    match = re.match(r'^(\d{8})_(\d{6})_(.+?)_([a-f0-9-]+)$', name)
                    if match:
                        date_str, time_str, stream_name, event_id = match.groups()
                        stream_name = stream_name.replace('_', ' ')
                        timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    else:
                        # Try format: StreamName_ID_YYYYMMDD_HHMMSS_type
                        match = re.match(r'^(.+?)_(\d+)_(\d{8})_(\d{6})_(.+)$', name)
                        if match:
                            stream_name, _, date_str, time_str, _ = match.groups()
                            stream_name = stream_name.replace('_', ' ')
                            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                        else:
                            # Unknown format, use file modification time
                            timestamp = datetime.fromtimestamp(clip_file.stat().st_mtime)
                            stream_name = name.split('_')[0] if '_' in name else "Unknown"
                    
                    # Check for matching thumbnail
                    thumb_filename = name + ".jpg"
                    thumb_path = THUMBNAILS_DIR / thumb_filename
                    if not thumb_path.exists():
                        thumb_filename = None
                    
                    # Get clip duration from file
                    cap = cv2.VideoCapture(str(clip_file))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    clip_duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    # Create event
                    db_event = Event(
                        stream_id=0,  # Unknown stream
                        stream_name=stream_name,
                        start_time=timestamp,
                        end_time=timestamp + timedelta(seconds=clip_duration) if clip_duration else None,
                        duration_seconds=clip_duration,
                        max_confidence=0.9,  # Assume high confidence (it was recorded for a reason)
                        avg_confidence=0.9,
                        min_confidence=0.9,
                        frame_count=int(frame_count) if frame_count else 1,
                        severity=AlertSeverity.HIGH,
                        status=EventStatus.PENDING,
                        clip_path=clip_filename,
                        clip_duration=clip_duration,
                        thumbnail_path=thumb_filename,
                    )
                    session.add(db_event)
                    imported += 1
                    logger.info(f"📥 Imported clip as event: {clip_filename} ({stream_name}, {timestamp})")
                    
                except Exception as e:
                    errors.append(f"{clip_filename}: {str(e)}")
                    logger.error(f"Failed to import clip {clip_filename}: {e}")
            
            await session.commit()
        
        return {
            "success": True,
            "message": f"Import complete: {imported} imported, {skipped} already existed",
            "imported": imported,
            "skipped": skipped,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Failed to import clips: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/clips/{clip_name}")
async def get_clip(clip_name: str, request: Request):
    """Serve a violence event clip with HTTP Range request support for browser video playback."""
    clip_path = CLIPS_DIR / clip_name
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    
    file_size = os.path.getsize(clip_path)
    range_header = request.headers.get("range")
    
    if range_header:
        # Parse Range header: "bytes=start-end"
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        def iter_file():
            with open(clip_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Disposition": f'inline; filename="{clip_name}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    
    # No range header — send full file
    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=clip_name,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/v1/clips/thumbnails/{thumb_name}")
async def get_thumbnail(thumb_name: str):
    """Serve a violence event thumbnail."""
    # Try thumbnails subdirectory first
    thumb_path = THUMBNAILS_DIR / thumb_name
    if not thumb_path.exists():
        # Fallback to clips directory (legacy location)
        thumb_path = CLIPS_DIR / thumb_name
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(
        thumb_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"}
    )


# ============== Face/Participant Image Routes ==============

FACE_PARTICIPANTS_DIR = CLIPS_DIR / "face_participants"
FACE_PARTICIPANTS_DIR.mkdir(exist_ok=True)


@app.post("/api/v1/faces/{event_id}/extract")
async def extract_faces_from_event(event_id: str):
    """Manually trigger face extraction for an event clip."""
    try:
        # Find the event in database first
        clip_filename = None
        
        try:
            event_id_int = int(event_id)
            async with async_session() as session:
                result = await session.execute(
                    select(Event).where(Event.id == event_id_int)
                )
                db_event = result.scalar_one_or_none()
                if db_event:
                    clip_filename = db_event.clip_path
        except (ValueError, Exception) as e:
            logger.warning(f"DB lookup failed for event {event_id}: {e}")
        
        # Fallback to in-memory events
        if not clip_filename:
            for e in stored_events:
                if e.get("event_id") == event_id or str(e.get("id")) == event_id:
                    clip_filename = e.get("clip_path")
                    break
        
        if not clip_filename:
            raise HTTPException(status_code=404, detail="Event not found")
        
        clip_path = CLIPS_DIR / clip_filename
        if not clip_path.exists():
            raise HTTPException(status_code=404, detail=f"Clip file not found: {clip_filename}")
        
        logger.info(f"🔍 Manual face extraction requested for event {event_id}, clip: {clip_path}")
        
        # Run face extraction
        face_extractor = get_face_extractor()
        faces = face_extractor.process_clip(str(clip_path), event_id)
        
        logger.info(f"✅ Extracted {len(faces)} faces for event {event_id}")
        
        return {
            "success": True,
            "data": {
                "event_id": event_id,
                "faces": faces,
                "count": len(faces),
                "clip_path": clip_filename
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face extraction failed for event {event_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/faces/{event_id}")
async def get_event_faces(event_id: str):
    """Get list of detected participant faces for an event."""
    try:
        face_extractor = get_face_extractor()
        faces = face_extractor.get_faces_for_event(event_id)
        return {
            "success": True,
            "data": {
                "event_id": event_id,
                "faces": faces,
                "count": len(faces)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get faces for event {event_id}: {e}")
        return {"success": True, "data": {"event_id": event_id, "faces": [], "count": 0}}


@app.get("/api/v1/faces/{event_id}/{face_name}")
async def get_face_image(event_id: str, face_name: str):
    """Serve a participant face image."""
    face_path = FACE_PARTICIPANTS_DIR / event_id / face_name
    if not face_path.exists():
        raise HTTPException(status_code=404, detail="Face image not found")
    
    return FileResponse(
        face_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"}
    )


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
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )

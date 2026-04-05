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
import socket
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
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
    Event, Stream as DBStream, EventStatus, AlertSeverity,
    VideoClip, ExtractedFace, InferenceLog
)
from app.config import settings

# Import storage service for clips and faces
from app.storage import get_storage_service

# Import encryption service for secure storage
from app.storage.encryption import get_encryption_service


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


# ============== Model Configuration ==============
MODEL_PATH = Path(__file__).parent / ".." / "ml-service" / "models" / "violence_model.onnx"
MODEL_PATH_LEGACY = Path(__file__).parent / ".." / "ml-service" / "models" / "violence_model_legacy.h5"
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

# Legacy storage paths (only used as fallback if encryption fails)
# DO NOT auto-create these directories - secure storage is the primary location
CLIPS_DIR = settings.clips_storage_path
THUMBNAILS_DIR = settings.thumbnails_storage_path

# Only create legacy directories when actually needed (in fallback code)
# NOT on startup - this prevents cluttering the project directory

STREAM_FPS = 30  # Default FPS for clip recording (overridden by actual measured FPS)


# ============== Helper Functions ==============

def verify_clip_file(file_path: Optional[str], check_encrypted: bool = True) -> Optional[str]:
    """
    Return the file path only if the file exists, otherwise return None.
    This ensures we don't return references to deleted files.
    
    Checks:
    1. Encrypted secure storage (primary)
    2. Legacy clips directory
    3. Absolute path
    4. Thumbnails directory
    """
    if not file_path:
        return None
    
    # First check encrypted storage (primary storage location for new files)
    if check_encrypted:
        try:
            encryption_service = get_encryption_service()
            # Check if it's a clip in encrypted storage
            if encryption_service.get_clip_info(file_path):
                return file_path
            # Check if it's a thumbnail in encrypted storage
            if file_path in encryption_service.manifest.get("thumbnails", {}):
                return file_path
        except Exception:
            pass  # Fall through to legacy checks
    
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
    """
    Loads and runs the violence detection model.
    
    Supports ONNX (primary, faster) and Keras/TensorFlow (fallback).
    ONNX provides 2-3x faster inference with lower memory usage.
    """
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.model_type = "none"  # "onnx", "keras", or "none"
        self._lock = threading.Lock()
        self._ort = None  # ONNX Runtime
        self._onnx_session = None
        self._input_name = None
        self._output_name = None
        self._load_model()
    
    def _load_model(self):
        """Load the model, preferring ONNX over Keras."""
        # Try ONNX first (faster)
        onnx_path = MODEL_PATH.resolve()
        if onnx_path.exists() and str(onnx_path).endswith('.onnx'):
            if self._load_onnx_model(onnx_path):
                return
        
        # Fallback to Keras/TensorFlow
        keras_path = MODEL_PATH_LEGACY.resolve()
        if keras_path.exists():
            self._load_keras_model(keras_path)
    
    def _load_onnx_model(self, model_path: Path) -> bool:
        """Load ONNX model with optimal execution provider."""
        try:
            import onnxruntime as ort
            self._ort = ort
            
            logger.info(f"Loading ONNX model from {model_path}...")
            
            # Get available providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            # Select providers (prefer GPU)
            providers = []
            execution_provider = "CPU"
            
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }))
                execution_provider = "CUDA"
            
            if 'DmlExecutionProvider' in available_providers:
                providers.append('DmlExecutionProvider')
                if execution_provider == "CPU":
                    execution_provider = "DirectML"
            
            providers.append('CPUExecutionProvider')
            
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = os.cpu_count() or 4
            sess_options.enable_mem_pattern = True
            
            # Create session
            self._onnx_session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            self._input_name = self._onnx_session.get_inputs()[0].name
            self._output_name = self._onnx_session.get_outputs()[0].name
            
            self.is_loaded = True
            self.model_type = "onnx"
            
            logger.info(f"✅ ONNX model loaded successfully ({execution_provider})")
            logger.info(f"  Input: {self._input_name} {self._onnx_session.get_inputs()[0].shape}")
            logger.info(f"  Output: {self._output_name}")
            
            # Warmup
            dummy = np.zeros((1, EXPECTED_FRAMES, *TARGET_SIZE, 3), dtype=np.float32)
            self._onnx_session.run([self._output_name], {self._input_name: dummy})
            logger.info("✅ ONNX model warmup complete")
            
            return True
            
        except ImportError:
            logger.warning("ONNX Runtime not installed, falling back to Keras")
            return False
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def _load_keras_model(self, model_path: Path):
        """Load the TensorFlow/Keras model (fallback)."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            tf.get_logger().setLevel('ERROR')
            
            try:
                self.model = keras.models.load_model(str(model_path), compile=False)
                self.is_loaded = True
                self.model_type = "keras"
                logger.info(f"✅ Loaded Keras model from {model_path}")
                
                # Warmup
                dummy = np.zeros((1, EXPECTED_FRAMES, *TARGET_SIZE, 3), dtype=np.float32)
                self.model.predict(dummy, verbose=0)
                logger.info("✅ Keras model warmup complete")
                return
                
            except Exception as e:
                logger.warning(f"Direct load failed: {str(e)[:80]}, trying architecture rebuild...")
            
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
            self.model_type = "keras"
            logger.info(f"✅ Loaded Keras model weights from {model_path}")
            
            # Warmup
            dummy = np.zeros((1, EXPECTED_FRAMES, *TARGET_SIZE, 3), dtype=np.float32)
            self.model.predict(dummy, verbose=0)
            logger.info("✅ Keras model warmup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Keras model: {e}")
            self.is_loaded = False
    
    def predict(self, frames: List[np.ndarray]) -> Optional[dict]:
        """Run prediction on frames using ONNX or Keras."""
        if not self.is_loaded:
            return None
        
        with self._lock:
            try:
                # Preprocess frames
                processed = self._preprocess(frames)
                
                start = time.time()
                
                if self.model_type == "onnx" and self._onnx_session:
                    # ONNX inference
                    outputs = self._onnx_session.run(
                        [self._output_name],
                        {self._input_name: processed}
                    )
                    prediction = outputs[0]
                else:
                    # Keras inference
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
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_type": self.model_type
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
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            resized = cv2.resize(rgb_frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
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
    face_paths: List[Any] = field(default_factory=list)  # Detected participant faces (paths or face data dicts)


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
        """
        Save video clip from collected frames with AES-256 encryption.
        
        Storage architecture:
        1. Writes temporary unencrypted clip to temp directory
        2. Extracts faces from the clip
        3. Encrypts and saves clip to secure storage
        4. Encrypts and saves thumbnail to secure storage
        5. Encrypts and saves faces to secure storage
        6. Stores metadata in PostgreSQL
        7. Broadcasts event completion via WebSocket
        8. Cleans up temporary files
        """
        import tempfile
        from app.storage.encryption import get_encryption_service
        
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
            actual_fps = STREAM_FPS  # default fallback
            if len(all_frames) >= 2:
                first_ts = all_frames[0][1]
                last_ts = all_frames[-1][1]
                elapsed = last_ts - first_ts
                if elapsed > 0:
                    actual_fps = len(all_frames) / elapsed
                    actual_fps = max(5.0, min(60.0, actual_fps))
                    logger.info(f"📊 Measured actual FPS: {actual_fps:.1f} (from {len(all_frames)} frames over {elapsed:.1f}s)")
            
            # Get frame dimensions
            first_frame = all_frames[0][0]
            height, width = first_frame.shape[:2]
            
            # Create temporary file for video encoding
            temp_dir = settings.temp_storage_path
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_clip_path = temp_dir / f"temp_{event.event_id[:8]}.mp4"
            
            # Write video to temp file using H.264 codec
            write_fps = int(round(actual_fps))
            video_written = False
            
            for codec in ['avc1', 'H264', 'x264', 'X264', 'mp4v']:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(str(temp_clip_path), fourcc, write_fps, (width, height))
                    
                    if out.isOpened():
                        for frame, _ in all_frames:
                            out.write(frame)
                        out.release()
                        
                        if temp_clip_path.exists() and temp_clip_path.stat().st_size > 1000:
                            video_written = True
                            break
                except Exception:
                    continue
            
            if not video_written:
                logger.error(f"❌ Failed to write temporary video file")
                return
            
            # Read the video file as bytes
            with open(temp_clip_path, 'rb') as f:
                video_data = f.read()
            
            clip_duration = len(all_frames) / actual_fps
            file_size = len(video_data)
            
            logger.info(f"📹 Temp clip created: {file_size/1024:.1f}KB, {clip_duration:.1f}s")
            
            # Create thumbnail from middle frame
            event_start_idx = len(event.pre_buffer_frames)
            event_end_idx = event_start_idx + len(event.event_frames)
            mid_idx = (event_start_idx + event_end_idx) // 2
            if mid_idx >= len(all_frames):
                mid_idx = len(all_frames) // 2
            
            thumbnail_data = None
            if mid_idx < len(all_frames):
                _, thumbnail_buffer = cv2.imencode('.jpg', all_frames[mid_idx][0], [cv2.IMWRITE_JPEG_QUALITY, 85])
                thumbnail_data = thumbnail_buffer.tobytes()
            
            # Extract faces from the temporary clip
            face_data_list = []
            try:
                face_extractor = get_face_extractor()
                face_data_list = face_extractor.process_clip(str(temp_clip_path), event.event_id)
                logger.info(f"👤 Extracted {len(face_data_list)} participant faces")
            except Exception as fe:
                logger.warning(f"Face extraction failed: {fe}")
            
            # Initialize encryption service and save encrypted files
            try:
                encryption_service = get_encryption_service()
                
                # Encrypt and save clip
                secure_clip_filename, clip_hash = encryption_service.encrypt_and_save_clip(
                    video_data=video_data,
                    event_id=event.event_id,
                    stream_id=event.stream_id,
                    stream_name=event.stream_name,
                    duration=clip_duration,
                    metadata={
                        "width": width,
                        "height": height,
                        "fps": actual_fps,
                        "frame_count": len(all_frames),
                        "max_score": event.max_score
                    }
                )
                event.clip_path = secure_clip_filename
                logger.info(f"🔐 Encrypted clip saved: {secure_clip_filename}")
                
                # Encrypt and save thumbnail
                secure_thumb_filename = None
                if thumbnail_data:
                    secure_thumb_filename = encryption_service.encrypt_and_save_thumbnail(
                        image_data=thumbnail_data,
                        event_id=event.event_id,
                        stream_id=event.stream_id
                    )
                    event.thumbnail_path = secure_thumb_filename
                    logger.info(f"🔐 Encrypted thumbnail saved: {secure_thumb_filename}")
                
                # Encrypt and save faces
                encrypted_face_refs = []
                for face_data in face_data_list:
                    if isinstance(face_data, dict) and 'image_data' in face_data:
                        secure_face_filename = encryption_service.encrypt_and_save_face(
                            image_data=face_data['image_data'],
                            event_id=event.event_id,
                            face_index=face_data.get('face_index', 0),
                            bbox=face_data.get('bbox')
                        )
                        encrypted_face_refs.append({
                            "secure_filename": secure_face_filename,
                            "face_index": face_data.get('face_index', 0),
                            "bbox": face_data.get('bbox')
                        })
                
                event.face_paths = encrypted_face_refs
                logger.info(f"🔐 Encrypted {len(encrypted_face_refs)} faces")
                
            except Exception as enc_error:
                logger.error(f"Encryption failed, falling back to unencrypted storage: {enc_error}")
                # Fallback: save unencrypted to legacy location
                # Create legacy directories ONLY when needed
                settings.ensure_legacy_directories()
                
                timestamp_str = event.start_time.strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c if c.isalnum() else "_" for c in event.stream_name)
                legacy_clip_filename = f"{timestamp_str}_{safe_name}_{event.event_id[:8]}.mp4"
                legacy_clip_path = CLIPS_DIR / legacy_clip_filename
                
                import shutil
                shutil.copy(temp_clip_path, legacy_clip_path)
                event.clip_path = legacy_clip_filename
                
                if thumbnail_data:
                    thumb_path = THUMBNAILS_DIR / f"{timestamp_str}_{safe_name}_{event.event_id[:8]}.jpg"
                    with open(thumb_path, 'wb') as f:
                        f.write(thumbnail_data)
                    event.thumbnail_path = f"thumbnails/{thumb_path.name}"
            
            # Clean up temporary file
            try:
                if temp_clip_path.exists():
                    temp_clip_path.unlink()
            except Exception:
                pass
            
            logger.info(f"✅ Saved encrypted clip for event {event.event_id} ({clip_duration:.1f}s, {file_size/1024:.1f}KB)")
            
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
    import base64
    
    avg_score = sum(event.frame_scores) / len(event.frame_scores) if event.frame_scores else 0
    
    # Process face data for JSON serialization
    # face_paths contains dicts with 'image_data' (bytes) - convert to base64 for WebSocket
    serializable_faces = []
    if event.face_paths:
        for face in event.face_paths:
            if isinstance(face, dict):
                face_copy = face.copy()
                # Convert bytes to base64 string for JSON serialization
                if 'image_data' in face_copy and isinstance(face_copy['image_data'], bytes):
                    face_copy['image_data'] = base64.b64encode(face_copy['image_data']).decode('utf-8')
                serializable_faces.append(face_copy)
            elif isinstance(face, str):
                # Already a path string, keep as-is
                serializable_faces.append(face)
    
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
        "face_paths": serializable_faces,  # Detected participant faces (base64 encoded)
        "participants_count": len(event.face_paths) if event.face_paths else 0,
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
    """
    Store event in PostgreSQL database along with VideoClip and ExtractedFace records.
    
    This function creates:
    1. Event record - main violence detection event
    2. VideoClip record - metadata about the saved clip file
    3. ExtractedFace records - one for each detected face
    """
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
            
            # Count faces if available
            face_paths = event.get("face_paths", [])
            person_count = len(face_paths) if face_paths else 0
            
            # Create Event record
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
                # Store secure encrypted storage IDs (same as clip_path/thumbnail_path for new events)
                secure_clip_id=event.get("clip_path"),
                secure_thumbnail_id=event.get("thumbnail_path"),
                person_count=person_count,
            )
            session.add(db_event)
            await session.flush()  # Flush to get the event ID
            
            event_id = db_event.id
            logger.info(f"📝 Event {event_id} stored in database: stream={event.get('stream_name')}, confidence={confidence:.1%}")
            
            # Create VideoClip record if clip_path exists
            clip_path = event.get("clip_path")
            if clip_path:
                try:
                    # Determine the full file path for the clip
                    encryption_service = get_encryption_service()
                    full_clip_path = str(encryption_service.clips_path / clip_path) if encryption_service else clip_path
                    
                    # Get file size if file exists
                    file_size = None
                    clip_file = Path(full_clip_path) if encryption_service else None
                    if clip_file and clip_file.exists():
                        file_size = clip_file.stat().st_size
                    
                    # Get thumbnail data if available (for inline storage in VideoClip)
                    thumbnail_data = None
                    thumbnail_path = event.get("thumbnail_path")
                    if thumbnail_path and encryption_service:
                        try:
                            # Decrypt thumbnail using the encryption service method
                            thumbnail_data = encryption_service.decrypt_thumbnail(thumbnail_path)
                        except Exception as thumb_err:
                            logger.warning(f"Could not read thumbnail for VideoClip: {thumb_err}")
                    
                    video_clip = VideoClip(
                        event_id=event_id,
                        stream_id=stream_id if stream_id > 0 else None,
                        filename=clip_path,
                        file_path=full_clip_path,
                        file_size_bytes=file_size,
                        duration_seconds=event.get("clip_duration"),
                        recorded_at=start_time,
                        thumbnail_data=thumbnail_data,
                        thumbnail_mime_type="image/jpeg" if thumbnail_data else None,
                    )
                    session.add(video_clip)
                    await session.flush()  # Flush to get clip ID for faces
                    
                    clip_id = video_clip.id
                    logger.info(f"🎬 VideoClip {clip_id} stored: {clip_path}")
                    
                    # Create ExtractedFace records for each detected face
                    if face_paths:
                        faces_stored = 0
                        for idx, face_data in enumerate(face_paths):
                            try:
                                # face_data can be a dict with image_data (base64 or bytes) and metadata
                                if isinstance(face_data, dict):
                                    # Get image data - may be base64 string or bytes
                                    image_data = face_data.get("image_data")
                                    if isinstance(image_data, str):
                                        # Base64 encoded string - decode it
                                        import base64
                                        image_data = base64.b64decode(image_data)
                                    
                                    if not image_data:
                                        # Try to read from secure filename if no inline data
                                        secure_filename = face_data.get("secure_filename")
                                        if secure_filename and encryption_service:
                                            try:
                                                # Use decrypt_face method instead of manual file access
                                                image_data = encryption_service.decrypt_face(secure_filename)
                                            except Exception as read_err:
                                                logger.warning(f"Could not read face file {secure_filename}: {read_err}")
                                    
                                    if not image_data:
                                        continue
                                    
                                    # Extract bounding box
                                    bbox = face_data.get("bbox", {})
                                    
                                    extracted_face = ExtractedFace(
                                        clip_id=clip_id,
                                        stream_id=stream_id if stream_id > 0 else None,
                                        event_id=event_id,
                                        image_data=image_data,
                                        image_mime_type="image/jpeg",
                                        image_size_bytes=len(image_data),
                                        face_index=face_data.get("face_index", idx),
                                        confidence=face_data.get("confidence"),
                                        bbox_x=bbox.get("x"),
                                        bbox_y=bbox.get("y"),
                                        bbox_width=bbox.get("width"),
                                        bbox_height=bbox.get("height"),
                                        frame_number=face_data.get("frame_number"),
                                        frame_timestamp_ms=face_data.get("frame_timestamp_ms"),
                                    )
                                    session.add(extracted_face)
                                    faces_stored += 1
                                    
                            except Exception as face_err:
                                logger.warning(f"Failed to store face {idx}: {face_err}")
                                continue
                        
                        if faces_stored > 0:
                            logger.info(f"👤 {faces_stored} ExtractedFace records stored for clip {clip_id}")
                    
                except Exception as clip_err:
                    logger.warning(f"Failed to create VideoClip record: {clip_err}")
                    import traceback
                    traceback.print_exc()
            
            await session.commit()
            logger.info(f"✅ Event {event_id} fully stored with clip and faces")
            
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


# ============== Sampled Inference Logging ==============

INFERENCE_LOG_SAMPLE_RATE = 10  # Log every 10th inference

async def log_inference_async(
    stream_id: int,
    violence_score: float,
    non_violence_score: float,
    inference_time_ms: Optional[float] = None,
    frame_number: Optional[int] = None
):
    """
    Store an inference result in the database (sampled).
    Called every N inferences to track trends without overwhelming the database.
    """
    try:
        async with async_session() as session:
            log_entry = InferenceLog(
                stream_id=stream_id,
                timestamp=datetime.utcnow(),
                violence_score=violence_score,
                non_violence_score=non_violence_score,
                inference_time_ms=inference_time_ms,
                frame_number=frame_number,
            )
            session.add(log_entry)
            await session.commit()
    except Exception as e:
        # Don't let logging failures affect inference
        logger.debug(f"Failed to log inference: {e}")


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
        
        # Inference counter for sampled logging
        self._inference_count = 0
        
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
                    
                    # Increment inference counter and log every Nth inference
                    self._inference_count += 1
                    if self._inference_count % INFERENCE_LOG_SAMPLE_RATE == 0:
                        # Log sampled inference to database (async, fire-and-forget)
                        if main_event_loop:
                            asyncio.run_coroutine_threadsafe(
                                log_inference_async(
                                    stream_id=self.id,
                                    violence_score=smoothed_score,
                                    non_violence_score=1.0 - smoothed_score,
                                    inference_time_ms=result.get("inference_time_ms"),
                                    frame_number=self.frame_count
                                ),
                                main_event_loop
                            )
                    
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
            
            # Use FFmpeg backend optimized for mobile hotspot/WiFi networks
            # TCP transport is more reliable over WiFi with packet loss
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'  # TCP for reliability over WiFi
                'fflags;nobuffer+discardcorrupt+fastseek|'  # No buffering at all
                'flags;low_delay|'  # Low delay decoding
                'framedrop;1|'  # Drop frames if behind
                'max_delay;500000|'  # Allow 500ms delay for network jitter
                'reorder_queue_size;10|'  # Small reorder queue for packet loss
                'analyzeduration;1000000|'  # 1s analyze (more time for slow networks)
                'probesize;500000|'  # 500KB probe (better stream detection)
                'flush_packets;1|'  # Flush immediately
                'avioflags;direct|'  # Direct I/O
                'stimeout;60000000|'  # 60s socket timeout
                'timeout;60000000'  # 60s connection timeout
            )
            
            self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Small buffer for stability over WiFi
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Native resolution for speed (no resize overhead)
            # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Match camera FPS
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Extended connection timeout for slow networks (15 seconds)
            connect_start = time.time()
            while time.time() - connect_start < 15:
                if self.capture.isOpened():
                    # Try to read a test frame to confirm connection
                    ret, _ = self.capture.read()
                    if ret:
                        self.is_connected = True
                        self.error = None
                        logger.info(f"Connected: {self.name}")
                        return
                time.sleep(0.2)  # Check every 200ms
            
            # Connection timed out
            self.is_connected = False
            self.error = "Connection timed out - check RTSP URL and network"
            logger.warning(f"Connection timeout: {self.name} - {self.url}")
            time.sleep(2)  # Wait before retry
                
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
    logger.info("Starting RTSP Service with Database Persistence...")
    
    # Initialize database
    await init_db()
    logger.info("PostgreSQL database initialized")
    
    # Initialize storage service
    storage = get_storage_service()
    logger.info(f"Storage service initialized - clips: {storage.clips_path}")
    
    # Initialize MongoDB for user authentication
    try:
        from app.auth import init_mongodb
        await init_mongodb()
        logger.info("MongoDB authentication service initialized")
    except Exception as e:
        logger.warning(f"MongoDB auth not available (optional): {e}")
    
    # Load streams from database
    await load_streams_from_db()
    
    # Initialize WebRTC streaming services (MediaMTX integration)
    try:
        from app.streaming import init_streaming_services
        await init_streaming_services()
        logger.info("WebRTC streaming services initialized")
    except Exception as e:
        logger.warning(f"WebRTC services not available: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    # Close MongoDB connection
    try:
        from app.auth import close_mongodb
        await close_mongodb()
    except Exception:
        pass
    
    try:
        from app.streaming import shutdown_streaming_services
        await shutdown_streaming_services()
    except Exception:
        pass
        
    stream_manager.shutdown()


app = FastAPI(
    title="SafeSight RTSP Stream Service",
    description="Ultra-low latency RTSP stream service with WebRTC delivery and AI violence detection",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include WebRTC streaming routes
try:
    from app.streaming import streaming_router
    app.include_router(streaming_router)
    logger.info("WebRTC streaming routes registered")
except ImportError as e:
    logger.warning(f"WebRTC routes not available: {e}")

# Include Authentication routes
try:
    from app.auth import auth_router
    app.include_router(auth_router, prefix="/api/v1")
    logger.info("Authentication routes registered at /api/v1/auth")
except ImportError as e:
    logger.warning(f"Auth routes not available: {e}")


# ============== API Models ==============

class StreamCreate(BaseModel):
    name: str = Field(..., description="Stream name")
    url: str = Field(..., description="RTSP URL")
    auto_start: bool = Field(default=True, description="Auto-start stream")


class StreamUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    location: Optional[str] = None
    custom_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


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


async def _update_stream(stream_id: int, request: StreamUpdate):
    """Update stream details. URL changes require a stream restart to take effect."""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    db_updates = {}

    if request.name is not None:
        stream.name = request.name
        db_updates["name"] = request.name
    if request.url is not None:
        stream.url = request.url
        db_updates["url"] = request.url
    if request.location is not None:
        db_updates["location"] = request.location
    if request.custom_threshold is not None:
        db_updates["custom_threshold"] = request.custom_threshold

    if db_updates:
        try:
            async with async_session() as session:
                await session.execute(
                    update(DBStream)
                    .where(DBStream.id == stream_id)
                    .values(**db_updates)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to persist stream update for {stream_id}: {e}")

    return {
        "success": True,
        "message": f"Stream {stream_id} updated. URL changes require restart.",
        "data": {
            "id": str(stream_id),
            "name": stream.name,
            "url": stream.url,
            "rtsp_url": stream.url,
            "location": request.location,
            "custom_threshold": request.custom_threshold,
        },
    }


@app.patch("/api/v1/streams/{stream_id}")
async def update_stream_patch(stream_id: int, request: StreamUpdate):
    return await _update_stream(stream_id, request)


@app.put("/api/v1/streams/{stream_id}")
async def update_stream_put(stream_id: int, request: StreamUpdate):
    return await _update_stream(stream_id, request)


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
            encryption_service = get_encryption_service()
            
            for e in events:
                clip_exists = verify_clip_file(e.clip_path)
                thumb_exists = verify_clip_file(e.thumbnail_path)
                
                # Get face paths from encrypted storage or database
                face_paths = []
                person_count = e.person_count or 0
                
                # Try encrypted storage first
                event_files = encryption_service.get_event_files(str(e.id))
                if event_files and event_files.get("faces"):
                    # Return secure filenames for frontend to use with /api/v1/faces/{event_id}/secure/{filename}
                    face_paths = event_files.get("faces", [])
                    person_count = len(face_paths) if not person_count else person_count
                else:
                    # Try database - get faces from ExtractedFace table
                    try:
                        faces_result = await session.execute(
                            select(ExtractedFace).where(ExtractedFace.event_id == e.id)
                        )
                        db_faces = faces_result.scalars().all()
                        if db_faces:
                            # Return face IDs for frontend to use with /api/v1/faces/{event_id}/image/{face_id}
                            face_paths = [f"db:{face.id}" for face in db_faces]
                            person_count = len(db_faces) if not person_count else person_count
                    except Exception:
                        pass
                
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
                    "person_count": person_count,
                    "participants_count": person_count,
                    "face_paths": face_paths,
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
                
                # Get face paths from encrypted storage or database
                face_paths = []
                person_count = e.person_count or 0
                encryption_service = get_encryption_service()
                
                # Try encrypted storage first
                event_files = encryption_service.get_event_files(str(e.id))
                if event_files and event_files.get("faces"):
                    face_paths = event_files.get("faces", [])
                    person_count = len(face_paths) if not person_count else person_count
                else:
                    # Try database
                    try:
                        faces_result = await session.execute(
                            select(ExtractedFace).where(ExtractedFace.event_id == e.id)
                        )
                        db_faces = faces_result.scalars().all()
                        if db_faces:
                            face_paths = [f"db:{face.id}" for face in db_faces]
                            person_count = len(db_faces) if not person_count else person_count
                    except Exception:
                        pass
                
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
                        "person_count": person_count,
                        "participants_count": person_count,
                        "face_paths": face_paths,
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


# ============== Video Clips Gallery API ==============

@app.get("/api/v1/video-clips")
async def get_video_clips(limit: int = 50, offset: int = 0, stream_id: Optional[int] = None):
    """
    Get all video clips from the database for the gallery view.
    
    Returns clip metadata with pagination support.
    Optionally filter by stream_id.
    """
    try:
        async with async_session() as session:
            from sqlalchemy import func
            
            # Build query
            query = select(VideoClip).order_by(VideoClip.recorded_at.desc())
            
            if stream_id is not None:
                query = query.where(VideoClip.stream_id == stream_id)
            
            # Get total count
            count_query = select(func.count(VideoClip.id))
            if stream_id is not None:
                count_query = count_query.where(VideoClip.stream_id == stream_id)
            count_result = await session.execute(count_query)
            total_count = count_result.scalar() or 0
            
            # Get clips with pagination
            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            clips = result.scalars().all()
            
            # Build response
            clip_list = []
            for clip in clips:
                # Get event info if available
                event_info = None
                if clip.event_id:
                    event_result = await session.execute(
                        select(Event).where(Event.id == clip.event_id)
                    )
                    event = event_result.scalar_one_or_none()
                    if event:
                        event_info = {
                            "id": event.id,
                            "severity": event.severity.value if event.severity else None,
                            "status": event.status.value if event.status else None,
                            "max_confidence": event.max_confidence,
                            "stream_name": event.stream_name
                        }
                
                clip_list.append({
                    "id": clip.id,
                    "event_id": clip.event_id,
                    "stream_id": clip.stream_id,
                    "filename": clip.filename,
                    "file_size_bytes": clip.file_size_bytes,
                    "duration_seconds": clip.duration_seconds,
                    "width": clip.width,
                    "height": clip.height,
                    "fps": clip.fps,
                    "frame_count": clip.frame_count,
                    "has_thumbnail": clip.thumbnail_data is not None,
                    "recorded_at": clip.recorded_at.isoformat() if clip.recorded_at else None,
                    "created_at": clip.created_at.isoformat() if clip.created_at else None,
                    "event": event_info
                })
            
            return {
                "success": True,
                "data": clip_list,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total_count
                }
            }
    except Exception as e:
        logger.error(f"Failed to get video clips: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/video-clips/{clip_id}")
async def get_video_clip(clip_id: int):
    """Get a specific video clip by ID."""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(VideoClip).where(VideoClip.id == clip_id)
            )
            clip = result.scalar_one_or_none()
            
            if not clip:
                raise HTTPException(status_code=404, detail="Clip not found")
            
            # Get event info
            event_info = None
            if clip.event_id:
                event_result = await session.execute(
                    select(Event).where(Event.id == clip.event_id)
                )
                event = event_result.scalar_one_or_none()
                if event:
                    event_info = {
                        "id": event.id,
                        "severity": event.severity.value if event.severity else None,
                        "status": event.status.value if event.status else None,
                        "max_confidence": event.max_confidence,
                        "stream_name": event.stream_name
                    }
            
            # Get face count
            face_count_result = await session.execute(
                select(ExtractedFace).where(ExtractedFace.clip_id == clip_id)
            )
            faces = face_count_result.scalars().all()
            
            return {
                "success": True,
                "data": {
                    "id": clip.id,
                    "event_id": clip.event_id,
                    "stream_id": clip.stream_id,
                    "filename": clip.filename,
                    "file_path": clip.file_path,
                    "file_size_bytes": clip.file_size_bytes,
                    "duration_seconds": clip.duration_seconds,
                    "width": clip.width,
                    "height": clip.height,
                    "fps": clip.fps,
                    "frame_count": clip.frame_count,
                    "has_thumbnail": clip.thumbnail_data is not None,
                    "recorded_at": clip.recorded_at.isoformat() if clip.recorded_at else None,
                    "created_at": clip.created_at.isoformat() if clip.created_at else None,
                    "event": event_info,
                    "face_count": len(faces)
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video clip: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/video-clips/{clip_id}/thumbnail")
async def get_video_clip_thumbnail(clip_id: int):
    """Get the thumbnail for a video clip (stored inline in DB)."""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(VideoClip).where(VideoClip.id == clip_id)
            )
            clip = result.scalar_one_or_none()
            
            if not clip or not clip.thumbnail_data:
                raise HTTPException(status_code=404, detail="Thumbnail not found")
            
            from fastapi.responses import Response
            return Response(
                content=clip.thumbnail_data,
                media_type=clip.thumbnail_mime_type or "image/jpeg"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get thumbnail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Extracted Faces Gallery API ==============

@app.get("/api/v1/extracted-faces")
async def get_extracted_faces(
    limit: int = 100, 
    offset: int = 0, 
    stream_id: Optional[int] = None,
    event_id: Optional[int] = None,
    clip_id: Optional[int] = None
):
    """
    Get all extracted faces from the database for the gallery view.
    
    Returns face metadata with pagination support.
    Optionally filter by stream_id, event_id, or clip_id.
    """
    try:
        async with async_session() as session:
            from sqlalchemy import func
            
            # Build query
            query = select(ExtractedFace).order_by(ExtractedFace.extracted_at.desc())
            
            if stream_id is not None:
                query = query.where(ExtractedFace.stream_id == stream_id)
            if event_id is not None:
                query = query.where(ExtractedFace.event_id == event_id)
            if clip_id is not None:
                query = query.where(ExtractedFace.clip_id == clip_id)
            
            # Get total count
            count_query = select(func.count(ExtractedFace.id))
            if stream_id is not None:
                count_query = count_query.where(ExtractedFace.stream_id == stream_id)
            if event_id is not None:
                count_query = count_query.where(ExtractedFace.event_id == event_id)
            if clip_id is not None:
                count_query = count_query.where(ExtractedFace.clip_id == clip_id)
            count_result = await session.execute(count_query)
            total_count = count_result.scalar() or 0
            
            # Get faces with pagination
            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            faces = result.scalars().all()
            
            # Build response
            face_list = []
            for face in faces:
                # Get event info if available
                event_info = None
                if face.event_id:
                    event_result = await session.execute(
                        select(Event).where(Event.id == face.event_id)
                    )
                    event = event_result.scalar_one_or_none()
                    if event:
                        event_info = {
                            "id": event.id,
                            "severity": event.severity.value if event.severity else None,
                            "status": event.status.value if event.status else None,
                            "stream_name": event.stream_name,
                            "start_time": event.start_time.isoformat() if event.start_time else None
                        }
                
                face_list.append({
                    "id": face.id,
                    "clip_id": face.clip_id,
                    "stream_id": face.stream_id,
                    "event_id": face.event_id,
                    "face_index": face.face_index,
                    "confidence": face.confidence,
                    "image_size_bytes": face.image_size_bytes,
                    "bbox": {
                        "x": face.bbox_x,
                        "y": face.bbox_y,
                        "width": face.bbox_width,
                        "height": face.bbox_height
                    } if face.bbox_x is not None else None,
                    "frame_number": face.frame_number,
                    "frame_timestamp_ms": face.frame_timestamp_ms,
                    "extracted_at": face.extracted_at.isoformat() if face.extracted_at else None,
                    "event": event_info
                })
            
            return {
                "success": True,
                "data": face_list,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total_count
                }
            }
    except Exception as e:
        logger.error(f"Failed to get extracted faces: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/extracted-faces/{face_id}")
async def get_extracted_face(face_id: int):
    """Get a specific extracted face by ID."""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(ExtractedFace).where(ExtractedFace.id == face_id)
            )
            face = result.scalar_one_or_none()
            
            if not face:
                raise HTTPException(status_code=404, detail="Face not found")
            
            # Get event info
            event_info = None
            if face.event_id:
                event_result = await session.execute(
                    select(Event).where(Event.id == face.event_id)
                )
                event = event_result.scalar_one_or_none()
                if event:
                    event_info = {
                        "id": event.id,
                        "severity": event.severity.value if event.severity else None,
                        "status": event.status.value if event.status else None,
                        "stream_name": event.stream_name,
                        "start_time": event.start_time.isoformat() if event.start_time else None
                    }
            
            return {
                "success": True,
                "data": {
                    "id": face.id,
                    "clip_id": face.clip_id,
                    "stream_id": face.stream_id,
                    "event_id": face.event_id,
                    "face_index": face.face_index,
                    "confidence": face.confidence,
                    "image_size_bytes": face.image_size_bytes,
                    "bbox": {
                        "x": face.bbox_x,
                        "y": face.bbox_y,
                        "width": face.bbox_width,
                        "height": face.bbox_height
                    } if face.bbox_x is not None else None,
                    "frame_number": face.frame_number,
                    "frame_timestamp_ms": face.frame_timestamp_ms,
                    "extracted_at": face.extracted_at.isoformat() if face.extracted_at else None,
                    "event": event_info
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get extracted face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/extracted-faces/{face_id}/image")
async def get_extracted_face_image(face_id: int):
    """Get the image for an extracted face (stored inline in DB)."""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(ExtractedFace).where(ExtractedFace.id == face_id)
            )
            face = result.scalar_one_or_none()
            
            if not face or not face.image_data:
                raise HTTPException(status_code=404, detail="Face image not found")
            
            from fastapi.responses import Response
            return Response(
                content=face.image_data,
                media_type=face.image_mime_type or "image/jpeg"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get face image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    Serve a violence event clip with HTTP Range request support for browser video playback.
    Automatically decrypts from secure storage if the clip is encrypted.
    """
    # First, try to decrypt from secure storage (primary storage location)
    try:
        encryption_service = get_encryption_service()
        clip_info = encryption_service.get_clip_info(clip_name)
        
        if clip_info:
            # Clip exists in encrypted storage - decrypt and serve
            decrypted_data = encryption_service.decrypt_clip(clip_name)
            if decrypted_data:
                file_size = len(decrypted_data)
                range_header = request.headers.get("range")
                
                if range_header:
                    # Parse Range header: "bytes=start-end"
                    range_str = range_header.replace("bytes=", "")
                    parts = range_str.split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if parts[1] else file_size - 1
                    end = min(end, file_size - 1)
                    content_length = end - start + 1
                    
                    import io
                    return StreamingResponse(
                        iter([decrypted_data[start:end + 1]]),
                        status_code=206,
                        media_type="video/mp4",
                        headers={
                            "Content-Range": f"bytes {start}-{end}/{file_size}",
                            "Accept-Ranges": "bytes",
                            "Content-Length": str(content_length),
                            "Content-Disposition": f'inline; filename="clip_{clip_name[:8]}.mp4"',
                            "Cache-Control": "private, no-store",
                        },
                    )
                
                # No range header — send full decrypted file
                import io
                return StreamingResponse(
                    io.BytesIO(decrypted_data),
                    media_type="video/mp4",
                    headers={
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(file_size),
                        "Content-Disposition": f'inline; filename="clip_{clip_name[:8]}.mp4"',
                        "Cache-Control": "private, no-store",
                    },
                )
    except Exception as e:
        logger.debug(f"Secure storage lookup failed for {clip_name}: {e}")
    
    # Fallback to legacy unencrypted storage
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
    """
    Serve a violence event thumbnail.
    Automatically decrypts from secure storage if the thumbnail is encrypted.
    """
    # First, try to decrypt from secure storage (primary storage location)
    try:
        encryption_service = get_encryption_service()
        
        # Check if this is an encrypted thumbnail
        thumb_info = encryption_service.manifest.get("thumbnails", {}).get(thumb_name)
        
        if thumb_info:
            # Thumbnail exists in encrypted storage - decrypt and serve
            decrypted_data = encryption_service.decrypt_thumbnail(thumb_name)
            if decrypted_data:
                import io
                return StreamingResponse(
                    io.BytesIO(decrypted_data),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "private, no-store"}
                )
    except Exception as e:
        logger.debug(f"Secure storage lookup failed for thumbnail {thumb_name}: {e}")
    
    # Fallback to legacy unencrypted storage
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


# ============== Secure Encrypted Storage API ==============

@app.get("/api/v1/secure/clips/{secure_filename}")
async def get_secure_clip(secure_filename: str, request: Request):
    """
    Serve an encrypted violence event clip with HTTP Range request support.
    Decrypts on-the-fly for streaming.
    """
    from app.storage.encryption import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        
        # Check if clip exists in manifest
        clip_info = encryption_service.get_clip_info(secure_filename)
        if not clip_info:
            # Fallback to legacy unencrypted storage
            legacy_path = CLIPS_DIR / secure_filename
            if legacy_path.exists():
                return FileResponse(
                    path=str(legacy_path),
                    media_type="video/mp4",
                    filename=f"clip_{secure_filename[:8]}.mp4"
                )
            raise HTTPException(status_code=404, detail="Clip not found")
        
        # Decrypt the clip
        video_data = encryption_service.decrypt_clip(secure_filename)
        if not video_data:
            raise HTTPException(status_code=500, detail="Failed to decrypt clip")
        
        file_size = len(video_data)
        range_header = request.headers.get("range")
        
        if range_header:
            # Parse Range header for partial content
            range_str = range_header.replace("bytes=", "")
            parts = range_str.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            content_length = end - start + 1
            
            def iter_chunk():
                yield video_data[start:end + 1]
            
            return StreamingResponse(
                iter_chunk(),
                status_code=206,
                media_type="video/mp4",
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                    "Cache-Control": "private, no-store",  # Don't cache decrypted content
                },
            )
        
        # No range header — send full decrypted file
        import io
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "private, no-store",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve secure clip {secure_filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve clip")


@app.get("/api/v1/secure/thumbnails/{secure_filename}")
async def get_secure_thumbnail(secure_filename: str):
    """Serve an encrypted thumbnail image."""
    from app.storage.encryption import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        
        # Decrypt thumbnail
        image_data = encryption_service.decrypt_thumbnail(secure_filename)
        if not image_data:
            # Fallback to legacy storage
            legacy_path = THUMBNAILS_DIR / secure_filename
            if legacy_path.exists():
                return FileResponse(legacy_path, media_type="image/jpeg")
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        import io
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, no-store"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve secure thumbnail {secure_filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve thumbnail")


@app.get("/api/v1/secure/faces/{secure_filename}")
async def get_secure_face(secure_filename: str):
    """Serve an encrypted face image."""
    from app.storage.encryption import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        
        # Decrypt face image
        image_data = encryption_service.decrypt_face(secure_filename)
        if not image_data:
            raise HTTPException(status_code=404, detail="Face image not found")
        
        import io
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, no-store"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve secure face {secure_filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve face")


@app.get("/api/v1/secure/events/{event_id}/files")
async def get_event_files(event_id: str):
    """Get all encrypted file references for an event."""
    from app.storage.encryption import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        event_files = encryption_service.get_event_files(event_id)
        
        if not event_files:
            raise HTTPException(status_code=404, detail="Event not found in secure storage")
        
        return {
            "event_id": event_id,
            "clips": event_files.get("clips", []),
            "thumbnails": event_files.get("thumbnails", []),
            "face_count": len(event_files.get("faces", [])),
            "created_at": event_files.get("created_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get event files for {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get event files")


@app.get("/api/v1/secure/manifest")
async def get_secure_manifest():
    """
    Get the encrypted manifest for web interface.
    This provides a summary of all stored events and their file references.
    """
    from app.storage.encryption import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        return encryption_service.get_manifest_json()
    except Exception as e:
        logger.error(f"Failed to get secure manifest: {e}")
        raise HTTPException(status_code=500, detail="Failed to get manifest")


@app.get("/api/v1/secure/storage-stats")
async def get_secure_storage_stats():
    """Get secure storage statistics."""
    from app.storage.encryption import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        return encryption_service.get_storage_stats()
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get storage stats")


@app.delete("/api/v1/secure/events/{event_id}")
async def delete_secure_event(event_id: str):
    """Delete all encrypted files for an event."""
    from app.storage.encryption import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        success = encryption_service.delete_event_files(event_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        
        return {"status": "deleted", "event_id": event_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete event files for {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete event")


# ============== Face/Participant Image Routes ==============

# Legacy face directory reference (DO NOT auto-create - use encrypted storage instead)
# Only used if legacy clip files still exist during migration
FACE_PARTICIPANTS_DIR = CLIPS_DIR / "face_participants"
# Note: Directory is NOT auto-created - encrypted storage is the primary location


@app.post("/api/v1/faces/{event_id}/extract")
async def extract_faces_from_event(event_id: str):
    """Manually trigger face extraction for an event clip and store in database."""
    try:
        # Find the event in database first
        clip_filename = None
        db_event_id = None
        stream_id = None
        clip_id = None
        secure_clip_filename = None  # For encrypted storage
        
        try:
            event_id_int = int(event_id)
            async with async_session() as session:
                result = await session.execute(
                    select(Event).where(Event.id == event_id_int)
                )
                db_event = result.scalar_one_or_none()
                if db_event:
                    clip_filename = db_event.clip_path
                    db_event_id = db_event.id
                    stream_id = db_event.stream_id
                    secure_clip_filename = db_event.secure_clip_id
                    
                    # Check if VideoClip record exists
                    clip_result = await session.execute(
                        select(VideoClip).where(VideoClip.event_id == db_event.id)
                    )
                    video_clip = clip_result.scalar_one_or_none()
                    if video_clip:
                        clip_id = video_clip.id
                        # Prefer persisted clip path/filename when legacy Event.clip_path is empty.
                        if not clip_filename:
                            clip_filename = video_clip.file_path or video_clip.filename
        except (ValueError, Exception) as e:
            logger.warning(f"DB lookup failed for event {event_id}: {e}")
        
        # Fallback to in-memory events
        if not clip_filename:
            for e in stored_events:
                if e.get("event_id") == event_id or str(e.get("id")) == event_id:
                    clip_filename = e.get("clip_path")
                    break
        
        # Try encrypted storage first
        encryption_svc = get_encryption_service()
        if not secure_clip_filename:
            event_files = encryption_svc.get_event_files(event_id)
            if event_files and event_files.get("clips"):
                # Use first clip from encrypted storage
                secure_clip_filename = event_files["clips"][0]

        if secure_clip_filename:
            logger.info(f"Found encrypted clip for event {event_id}: {secure_clip_filename}")
            
            # Decrypt to temp file for face extraction
            decrypted_data = encryption_svc.decrypt_clip(secure_clip_filename)
            if decrypted_data:
                import tempfile
                temp_clip_path = Path(settings.temp_storage_path) / f"temp_extract_{event_id}.mp4"
                temp_clip_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_clip_path, 'wb') as f:
                    f.write(decrypted_data)
                
                logger.info(f"Decrypted clip to temp file for face extraction: {temp_clip_path}")
                
                # Run face extraction
                face_extractor = get_face_extractor()
                faces_data = face_extractor.process_clip(str(temp_clip_path), event_id)
                
                # Clean up temp file
                try:
                    temp_clip_path.unlink()
                except Exception:
                    pass
                
                # Save faces to encrypted storage and database
                saved_faces = []
                for i, face in enumerate(faces_data):
                    if face.get("image_data"):
                        # Save to encrypted storage
                        face_secure_id = encryption_svc.encrypt_and_save_face(
                            face["image_data"],
                            event_id,
                            i,
                            (face.get("bbox_x"), face.get("bbox_y"), 
                             face.get("bbox_width"), face.get("bbox_height"))
                        )
                        face["secure_filename"] = face_secure_id
                        saved_faces.append(face)
                
                # Also save to database if we have IDs
                if db_event_id and clip_id:
                    try:
                        storage = get_storage_service()
                        for face in faces_data:
                            face['clip_id'] = clip_id
                            face['event_id'] = db_event_id
                            face['stream_id'] = stream_id
                        
                        db_faces = await storage.save_faces_batch(
                            faces_data, clip_id, db_event_id, stream_id
                        )
                        logger.info(f"Saved {len(db_faces)} faces to database for event {event_id}")
                        
                        # Update event person_count
                        async with async_session() as session:
                            async with session.begin():
                                event = await session.get(Event, db_event_id)
                                if event:
                                    event.person_count = len(saved_faces)
                            await session.commit()
                    except Exception as db_err:
                        logger.warning(f"Failed to save faces to database: {db_err}")
                
                logger.info(f"Extracted {len(saved_faces)} faces for event {event_id}")
                
                return {
                    "success": True,
                    "data": {
                        "event_id": event_id,
                        "faces_count": len(saved_faces),
                        "stored_in_db": bool(db_event_id and clip_id),
                        "storage": "encrypted"
                    }
                }
        
        # Fallback to legacy filesystem (old clips before encryption)
        if not clip_filename:
            raise HTTPException(status_code=404, detail="Event not found")
        
        clip_path = CLIPS_DIR / clip_filename
        if not clip_path.exists():
            raise HTTPException(status_code=404, detail=f"Clip file not found: {clip_filename}")
        
        logger.info(f"Manual face extraction requested for event {event_id}, clip: {clip_path}")
        
        # Run face extraction
        face_extractor = get_face_extractor()
        faces_data = face_extractor.process_clip(str(clip_path), event_id)
        
        # If we have database IDs, save faces to PostgreSQL
        if db_event_id and clip_id:
            try:
                storage = get_storage_service()
                for face in faces_data:
                    face['clip_id'] = clip_id
                    face['event_id'] = db_event_id
                    face['stream_id'] = stream_id
                
                saved_faces = await storage.save_faces_batch(
                    faces_data, clip_id, db_event_id, stream_id
                )
                logger.info(f"Saved {len(saved_faces)} faces to database for event {event_id}")
                
                # Update event person_count
                async with async_session() as session:
                    async with session.begin():
                        event = await session.get(Event, db_event_id)
                        if event:
                            event.person_count = len(saved_faces)
                    await session.commit()
            except Exception as db_err:
                logger.warning(f"Failed to save faces to database: {db_err}")
        
        logger.info(f"Extracted {len(faces_data)} faces for event {event_id}")
        
        return {
            "success": True,
            "data": {
                "event_id": event_id,
                "faces_count": len(faces_data),
                "stored_in_db": bool(db_event_id and clip_id),
                "storage": "legacy"
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
    """
    Get list of detected participant faces for an event.
    Checks encrypted storage first, then database, then legacy filesystem.
    """
    try:
        # First, try encrypted storage (primary storage location)
        try:
            encryption_service = get_encryption_service()
            event_files = encryption_service.get_event_files(event_id)
            
            if event_files and event_files.get("faces"):
                face_list = []
                for i, secure_filename in enumerate(event_files.get("faces", [])):
                    face_info = encryption_service.manifest.get("faces", {}).get(secure_filename, {})
                    face_list.append({
                        "id": secure_filename,
                        "face_index": face_info.get("face_index", i),
                        "url": f"/api/v1/faces/{event_id}/secure/{secure_filename}",
                        "secure_filename": secure_filename,
                        "bbox": {
                            "x": face_info.get("bbox", [None])[0],
                            "y": face_info.get("bbox", [None, None])[1] if face_info.get("bbox") and len(face_info.get("bbox", [])) > 1 else None,
                            "width": face_info.get("bbox", [None, None, None])[2] if face_info.get("bbox") and len(face_info.get("bbox", [])) > 2 else None,
                            "height": face_info.get("bbox", [None, None, None, None])[3] if face_info.get("bbox") and len(face_info.get("bbox", [])) > 3 else None,
                        } if face_info.get("bbox") else None,
                    })
                return {
                    "success": True,
                    "data": {
                        "event_id": event_id,
                        "faces": face_list,
                        "count": len(face_list),
                        "source": "encrypted"
                    }
                }
        except Exception as e:
            logger.debug(f"Encrypted storage face lookup failed: {e}")
        
        # Try to get faces from database
        try:
            event_id_int = int(event_id)
            storage = get_storage_service()
            faces = await storage.get_faces_by_event(event_id_int)
            
            if faces:
                face_list = [
                    {
                        "id": face.id,
                        "face_index": face.face_index,
                        "url": f"/api/v1/faces/{event_id}/image/{face.id}",
                        "bbox": {
                            "x": face.bbox_x,
                            "y": face.bbox_y,
                            "width": face.bbox_width,
                            "height": face.bbox_height
                        } if face.bbox_x is not None else None,
                        "frame_number": face.frame_number,
                        "confidence": face.confidence
                    }
                    for face in faces
                ]
                return {
                    "success": True,
                    "data": {
                        "event_id": event_id,
                        "faces": face_list,
                        "count": len(faces),
                        "source": "database"
                    }
                }
        except (ValueError, Exception) as e:
            logger.debug(f"DB face lookup failed: {e}")
        
        # Fallback to legacy filesystem
        face_dir = FACE_PARTICIPANTS_DIR / event_id
        if face_dir.exists():
            faces = sorted([f.name for f in face_dir.glob("*.jpg")])
            face_list = [
                {
                    "face_index": i,
                    "url": f"/api/v1/faces/{event_id}/{name}",
                    "filename": name
                }
                for i, name in enumerate(faces)
            ]
            return {
                "success": True,
                "data": {
                    "event_id": event_id,
                    "faces": face_list,
                    "count": len(faces),
                    "source": "filesystem"
                }
            }
        
        return {"success": True, "data": {"event_id": event_id, "faces": [], "count": 0}}
    except Exception as e:
        logger.error(f"Failed to get faces for event {event_id}: {e}")
        return {"success": True, "data": {"event_id": event_id, "faces": [], "count": 0}}


@app.get("/api/v1/faces/{event_id}/image/{face_id}")
async def get_face_image_by_id(event_id: str, face_id: int):
    """Serve a participant face image from database."""
    try:
        storage = get_storage_service()
        face = await storage.get_face_by_id(face_id)
        
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")
        
        return StreamingResponse(
            iter([face.image_data]),
            media_type=face.image_mime_type or "image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get face image {face_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/faces/{event_id}/{face_name}")
async def get_face_image(event_id: str, face_name: str):
    """Serve a participant face image from legacy filesystem."""
    face_path = FACE_PARTICIPANTS_DIR / event_id / face_name
    if not face_path.exists():
        raise HTTPException(status_code=404, detail="Face image not found")
    
    return FileResponse(
        face_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"}
    )


@app.get("/api/v1/faces/{event_id}/secure/{secure_filename}")
async def get_secure_face_image(event_id: str, secure_filename: str):
    """Serve a participant face image from encrypted storage."""
    try:
        encryption_service = get_encryption_service()
        
        # Decrypt face image
        image_data = encryption_service.decrypt_face(secure_filename)
        if not image_data:
            raise HTTPException(status_code=404, detail="Face image not found")
        
        import io
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, no-store"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve secure face {secure_filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve face")


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
    def _is_port_available(host: str, port: int) -> bool:
        """Check if a TCP port can be bound on the target host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return True
            except OSError:
                return False

    def _find_available_port(host: str, preferred_port: int, max_attempts: int = 20) -> Optional[int]:
        """Return preferred port if available, otherwise first free port in the scan range."""
        if _is_port_available(host, preferred_port):
            return preferred_port

        for offset in range(1, max_attempts + 1):
            candidate = preferred_port + offset
            if _is_port_available(host, candidate):
                return candidate

        return None

    host = settings.host
    preferred_port = settings.port
    selected_port = _find_available_port(host, preferred_port)

    if selected_port is None:
        logger.error(
            f"No free port found in range {preferred_port}-{preferred_port + 20}. "
            "Set PORT in .env to an available value."
        )
        raise SystemExit(1)

    if selected_port != preferred_port:
        logger.warning(
            f"Port {preferred_port} is already in use. Starting RTSP service on port {selected_port} instead."
        )

    uvicorn.run(
        "main:app",
        host=host,
        port=selected_port,
        reload=False,
        log_level=settings.log_level.lower()
    )

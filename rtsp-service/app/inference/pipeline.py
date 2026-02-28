"""
RTSP Live Stream Service - Inference Pipeline
==============================================
Sliding window inference with continuous scoring
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import numpy as np
import cv2
import aiohttp
from loguru import logger

from app.config import settings
from app.stream.ingestion import FrameData, StreamIngestion
from app.utils.motion_analysis import CameraShakeDetector, ScoreStabilizer, MotionAnalysis


@dataclass
class InferenceResult:
    """Result of a single inference."""
    violence_score: float
    non_violence_score: float
    timestamp: datetime
    inference_time_ms: float
    frame_count: int
    window_start: datetime
    window_end: datetime
    stream_id: int
    # Motion analysis fields
    is_camera_shake: bool = False
    shake_score: float = 0.0
    stabilized_score: float = 0.0
    is_confirmed: bool = False  # True when violence sustained for 4-5 seconds
    raw_score: float = 0.0  # Original unmodified score
    is_stable: bool = True  # True when camera is stable (no global motion)
    
    @property
    def is_violent(self) -> bool:
        # CRITICAL: Only detect violence when camera is STABLE
        # Any camera movement = automatic rejection
        if self.is_camera_shake:
            return False  # Never trigger during camera shake
        if not self.is_stable:
            return False  # Never trigger when scene is unstable
        if self.is_confirmed:
            return True  # Sustained violence confirmed
        # For stable scenes only
        return self.stabilized_score >= settings.violence_threshold
    
    @property
    def classification(self) -> str:
        return "violence" if self.is_violent else "non-violence"
    
    @property
    def confidence(self) -> float:
        return max(self.violence_score, self.non_violence_score)


@dataclass
class SlidingWindowState:
    """State for sliding window inference."""
    stream_id: int
    recent_scores: List[float] = field(default_factory=list)
    last_inference_time: Optional[datetime] = None
    consecutive_violent_frames: int = 0
    is_in_event: bool = False
    event_start_time: Optional[datetime] = None
    event_scores: List[float] = field(default_factory=list)


class LocalModelInference:
    """
    Local model inference using the Keras model directly with GPU optimization.
    Falls back to ML service API if local model not available.
    
    The model is a full end-to-end MobileNetV2+LSTM architecture:
    - Input: (batch, 16 frames, 224, 224, 3) - raw RGB frames
    - TimeDistributed(MobileNetV2) -> feature extraction
    - TimeDistributed(GlobalAveragePooling2D) -> (batch, 16, 1280)
    - LSTM(64) -> (batch, 64)
    - Dense(64, relu)
    - Dense(1, sigmoid): violence probability
    """
    
    # Model expects 16 frames of 224x224 RGB
    EXPECTED_FRAMES = 16
    TARGET_SIZE = (224, 224)
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.model_path
        self.model = None
        self.use_local = False
        self._tf = None
        self._warmup_done = False
        self._load_model()
    
    def _load_model(self):
        """Load the full MobileNetV2+LSTM model with GPU optimization."""
        try:
            if not self.model_path or not Path(self.model_path).exists():
                logger.warning(f"Local model not found at {self.model_path}, will use ML service API")
                return
                
            import tensorflow as tf
            self._tf = tf
            from tensorflow import keras
            
            # Log GPU status
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"GPU(s) available for inference: {len(gpus)}")
                for gpu in gpus:
                    logger.info(f"  - {gpu.name}")
            else:
                logger.warning("No GPU detected - inference will be slower")
            
            # Load the complete model directly (includes MobileNetV2 + LSTM)
            logger.info(f"Loading full model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path, compile=False)
            
            # Get expected frames from model input shape
            input_shape = self.model.input_shape
            if input_shape and len(input_shape) == 5 and input_shape[1]:
                self.EXPECTED_FRAMES = input_shape[1]
            
            logger.info(f"Model loaded successfully")
            logger.info(f"  Input shape: {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")
            logger.info(f"  Expected frames: {self.EXPECTED_FRAMES}")
            
            self.use_local = True
            
            # Warmup
            self._warmup_model()
                
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            import traceback
            traceback.print_exc()
            self.use_local = False
    
    def _warmup_model(self):
        """Warmup the model with a dummy inference to optimize GPU memory allocation."""
        if self._warmup_done or self.model is None:
            return
        
        try:
            logger.info("Warming up model for GPU optimization...")
            
            dummy_input = np.zeros((1, self.EXPECTED_FRAMES, 224, 224, 3), dtype=np.float32)
            for _ in range(3):
                _ = self.model.predict(dummy_input, verbose=0)
            
            self._warmup_done = True
            logger.info("Model warmup complete - GPU memory allocated")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def preprocess_frames(self, frames: List[np.ndarray], target_size: tuple = None) -> np.ndarray:
        """
        Preprocess raw BGR frames for model input.
        
        Input: List of BGR frames (OpenCV format)
        Output: (1, EXPECTED_FRAMES, 224, 224, 3) - float32 RGB frames
        """
        target_size = target_size or self.TARGET_SIZE
        
        # Ensure we have exactly EXPECTED_FRAMES frames
        if len(frames) < self.EXPECTED_FRAMES:
            # Pad with repeated last frame
            frames = list(frames) + [frames[-1]] * (self.EXPECTED_FRAMES - len(frames))
        elif len(frames) > self.EXPECTED_FRAMES:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.EXPECTED_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Preprocess frames
        processed = []
        for frame in frames:
            # Convert BGR (OpenCV default) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to 224x224
            resized = cv2.resize(rgb_frame, target_size, interpolation=cv2.INTER_AREA)
            # Convert to float32 and normalize to [0, 1]
            # The model was trained with rescale=1./255 normalization
            img = resized.astype(np.float32) / 255.0
            processed.append(img)
        
        # Stack frames: (EXPECTED_FRAMES, 224, 224, 3)
        batch = np.stack(processed, axis=0)
        
        # Add batch dimension: (1, EXPECTED_FRAMES, 224, 224, 3)
        return np.expand_dims(batch, axis=0)
    
    def predict(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Run inference on frames with GPU optimization.
        Uses model.__call__ for lower latency than model.predict.
        """
        if not self.use_local or self.model is None:
            raise RuntimeError("Local model not available")
        
        # Preprocess frames into model input format
        input_data = self.preprocess_frames(frames)
        
        # Run inference with optimized call
        start_time = time.time()
        
        # Use model() directly instead of model.predict() for faster GPU inference
        if self._tf is not None:
            # Convert to tensor for faster GPU transfer
            input_tensor = self._tf.constant(input_data, dtype=self._tf.float32)
            predictions = self.model(input_tensor, training=False)
            predictions = predictions.numpy()
        else:
            predictions = self.model.predict(input_data, verbose=0)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse output
        if predictions.shape[-1] == 2:
            # Two outputs: [violence, non_violence]
            violence_score = float(predictions[0][0])
            non_violence_score = float(predictions[0][1])
        else:
            # Single output (violence probability via sigmoid)
            violence_score = float(predictions[0][0])
            non_violence_score = 1.0 - violence_score
        
        return {
            "violence_score": violence_score,
            "non_violence_score": non_violence_score,
            "inference_time_ms": inference_time
        }


class MLServiceInference:
    """Inference using the remote ML service API."""
    
    def __init__(self, base_url: str = None, timeout: int = None):
        self.base_url = base_url or settings.ml_service_url
        self.timeout = timeout or settings.ml_service_timeout
    
    async def predict_from_frames(
        self,
        frames: List[np.ndarray],
        stream_id: int
    ) -> Dict[str, Any]:
        """
        Send frames to ML service for inference.
        Creates a temporary video file from frames.
        """
        temp_video_path = None
        
        try:
            # Create temporary video from frames
            temp_video_path = self._create_temp_video(frames)
            
            if not temp_video_path:
                raise RuntimeError("Failed to create temporary video")
            
            # Send to ML service
            result = await self._send_to_ml_service(temp_video_path)
            return result
            
        finally:
            # Cleanup temp file
            if temp_video_path and Path(temp_video_path).exists():
                try:
                    Path(temp_video_path).unlink()
                except:
                    pass
    
    def _create_temp_video(self, frames: List[np.ndarray], fps: float = 15.0) -> Optional[str]:
        """Create a temporary video file from frames."""
        if not frames:
            return None
        
        try:
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                # Fallback for systems without H.264 support
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create temp video: {e}")
            return None
    
    async def _send_to_ml_service(self, video_path: str) -> Dict[str, Any]:
        """Send video to ML service API."""
        url = f"{self.base_url}/inference/predict-upload"
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                with open(video_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field(
                        'video',
                        f,
                        filename='inference.mp4',
                        content_type='video/mp4'
                    )
                    
                    async with session.post(url, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            return {
                                "violence_score": result.get("probabilities", {}).get("violence", 0.0),
                                "non_violence_score": result.get("probabilities", {}).get("nonViolence", 1.0),
                                "inference_time_ms": result.get("metrics", {}).get("inferenceTime", 0) * 1000
                            }
                        else:
                            text = await response.text()
                            raise RuntimeError(f"ML service error: {response.status} - {text}")
                            
        except asyncio.TimeoutError:
            raise RuntimeError("ML service timeout")
        except Exception as e:
            logger.error(f"ML service request failed: {e}")
            raise


class InferencePipeline:
    """
    CCTV-style continuous inference pipeline.
    
    Checks every frame by using a sliding window of the last 16 consecutive 
    frames. At each inference cycle, the window has advanced by a few frames
    (depending on FPS and inference interval), giving continuous, overlapping
    coverage like a professional CCTV system.
    
    At 30fps camera + 100ms inference interval:
    - ~10 inferences/second
    - Window advances ~3 frames between cycles 
    - Every single frame participates in ~5 inference windows
    - 16 consecutive frames = ~0.53s of video per inference
    """
    
    def __init__(
        self,
        stream: StreamIngestion,
        on_result: Optional[Callable[[InferenceResult], None]] = None,
        use_local_model: bool = True
    ):
        self.stream = stream
        self.on_result = on_result
        
        # Initialize inference backend
        if use_local_model:
            try:
                self.local_inference = LocalModelInference()
                self.use_local = self.local_inference.use_local
            except:
                self.use_local = False
        else:
            self.use_local = False
        
        if not self.use_local:
            self.ml_service = MLServiceInference()
        
        # State
        self.state = SlidingWindowState(stream_id=stream.config.id)
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        
        # Scoring history for smoothing
        self.score_history: List[float] = []
        self.max_history_size = 20  # Keep more history for CCTV-style smoothing
        
        # Track last processed frame to detect new frames
        self._last_frame_number: int = -1
        
        # Camera shake detection and score stabilization
        # These prevent false positives from camera shake/rapid motion
        self.shake_detector = CameraShakeDetector()
        self.score_stabilizer = ScoreStabilizer(
            confirmation_window_seconds=settings.shake_confirmation_seconds,  # Require 4 seconds sustained detection
            inference_rate_hz=1000.0 / settings.inference_interval_ms,  # Match inference rate
            min_confirmations=12,  # Need 12+ high scores in the window
            decay_factor=0.85,
            shake_penalty=settings.shake_score_penalty  # Reduce score during shake
        )
        
        # Timestamp for stabilizer
        self._start_time = time.time()
    
    async def _inference_loop(self):
        """
        CCTV-style continuous inference loop.
        
        Runs at high frequency, always using the LAST 16 consecutive frames
        from the camera. This ensures every frame is checked as part of at
        least one inference window, mimicking how professional CCTV analytics
        continuously monitor the feed.
        """
        required_frames = LocalModelInference.EXPECTED_FRAMES  # 16
        interval_seconds = settings.inference_interval_ms / 1000.0
        
        logger.info(
            f"🎬 Starting CCTV-style continuous inference for [{self.stream.config.name}] "
            f"(interval: {settings.inference_interval_ms}ms = "
            f"{1000 / settings.inference_interval_ms:.1f} checks/sec, "
            f"window: {required_frames} consecutive frames)"
        )
        
        while self.is_running:
            try:
                # Check if stream is connected
                if not self.stream.is_connected:
                    await asyncio.sleep(0.5)
                    continue
                
                # Get the LAST 16 CONSECUTIVE frames — no sampling, no gaps
                frames = self.stream.get_consecutive_frames(required_frames)
                
                if len(frames) < required_frames:
                    # Not enough frames accumulated yet (camera just started)
                    await asyncio.sleep(interval_seconds)
                    continue
                
                # Skip if no new frames since last inference (avoid redundant work)
                latest_frame_num = frames[-1].frame_number
                if latest_frame_num == self._last_frame_number:
                    await asyncio.sleep(0.01)  # Brief wait for new frame
                    continue
                self._last_frame_number = latest_frame_num
                
                # Time the full cycle: inference + sleep = constant interval
                cycle_start = asyncio.get_event_loop().time()
                
                # Run inference on consecutive frames
                result = await self._run_inference(frames)
                
                if result and self.on_result:
                    self.on_result(result)
                
                # Sleep for remaining interval time
                elapsed = asyncio.get_event_loop().time() - cycle_start
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Inference error: {e}")
                await asyncio.sleep(interval_seconds)
        
        logger.info(f"Inference pipeline stopped for stream {self.stream.config.name}")
    
    async def _run_inference(self, frames: List[FrameData]) -> Optional[InferenceResult]:
        """Run inference on consecutive frames (CCTV-style) with shake detection."""
        if not frames:
            return None
        
        start_time = time.time()
        
        try:
            # Extract numpy arrays from frame data
            frame_arrays = [f.frame for f in frames]
            
            # Step 1: Analyze frames for camera shake and suspicious motion
            motion_analysis = self.shake_detector.analyze_frames(frame_arrays)
            is_shake = motion_analysis.is_camera_shake
            shake_score = motion_analysis.shake_score
            is_static = motion_analysis.is_static_scene
            is_suspicious = motion_analysis.is_suspicious_motion
            is_stable = motion_analysis.is_stable  # Camera stability status
            
            # Run inference
            if self.use_local and self.local_inference:
                result_data = self.local_inference.predict(frame_arrays)
            else:
                result_data = await self.ml_service.predict_from_frames(
                    frame_arrays,
                    self.stream.config.id
                )
            
            inference_time = (time.time() - start_time) * 1000
            
            # Get raw violence score
            raw_violence_score = result_data["violence_score"]
            
            # Step 2: Apply score stabilization with full motion analysis
            # CRITICAL: Pass is_stable - scores are ZEROED when camera is moving
            current_timestamp = time.time() - self._start_time
            stabilized_score, is_confirmed = self.score_stabilizer.add_score(
                raw_score=raw_violence_score,
                timestamp=current_timestamp,
                is_camera_shake=is_shake,
                shake_score=shake_score,
                is_static_scene=is_static,
                is_suspicious_motion=is_suspicious,
                is_stable=is_stable
            )
            
            # Update score history (use stabilized score)
            self.score_history.append(stabilized_score)
            if len(self.score_history) > self.max_history_size:
                self.score_history.pop(0)
            
            # Compute window span from actual frame timestamps
            window_span_ms = (frames[-1].timestamp - frames[0].timestamp).total_seconds() * 1000
            
            # Check if this is a problematic frame (any camera motion)
            is_problematic = is_shake or is_static or is_suspicious or not is_stable
            
            # Create result with motion analysis data
            result = InferenceResult(
                violence_score=stabilized_score,  # Use stabilized score for detection
                non_violence_score=1.0 - stabilized_score,
                timestamp=datetime.utcnow(),
                inference_time_ms=result_data.get("inference_time_ms", inference_time),
                frame_count=len(frames),
                window_start=frames[0].timestamp,
                window_end=frames[-1].timestamp,
                stream_id=self.stream.config.id,
                is_camera_shake=is_shake,
                shake_score=shake_score,
                stabilized_score=stabilized_score,
                is_confirmed=is_confirmed,
                raw_score=raw_violence_score,
                is_stable=is_stable
            )
            
            # Update state
            self.state.last_inference_time = result.timestamp
            self.state.recent_scores.append(stabilized_score)
            if len(self.state.recent_scores) > 30:
                self.state.recent_scores.pop(0)
            
            # Enhanced logging with full motion analysis info
            score_count = len(self.state.recent_scores)
            
            # Build status indicators
            status_flags = []
            if is_shake:
                status_flags.append("📳SHAKE")
            if is_static:
                status_flags.append("🔲STATIC")
            if is_suspicious:
                status_flags.append("⚠️SUSPICIOUS")
            if not is_stable:
                status_flags.append("🔄UNSTABLE")
            if is_confirmed:
                status_flags.append("✅CONFIRMED")
            status_str = " ".join(status_flags) if status_flags else "✓STABLE"
            
            if stabilized_score >= settings.violence_threshold and is_stable and not is_shake:
                avg = sum(self.state.recent_scores[-5:]) / min(5, len(self.state.recent_scores))
                logger.warning(
                    f"🔴 VIOLENT [{self.stream.config.name}] "
                    f"raw={raw_violence_score:.1%} stab={stabilized_score:.1%} avg5={avg:.1%} "
                    f"frames={len(frames)} span={window_span_ms:.0f}ms "
                    f"sim={motion_analysis.frame_similarity:.1%} {status_str} "
                    f"({inference_time:.0f}ms)"
                )
            elif is_problematic and raw_violence_score >= settings.violence_threshold:
                # Log when high raw scores are being suppressed
                logger.info(
                    f"🚫 SUPPRESSED [{self.stream.config.name}] "
                    f"raw={raw_violence_score:.1%} → stab={stabilized_score:.1%} "
                    f"mag={motion_analysis.global_motion_magnitude:.1f} "
                    f"uniform={motion_analysis.motion_uniformity:.1%} "
                    f"stability={motion_analysis.stability_duration:.1f}s {status_str} "
                    f"({inference_time:.0f}ms)"
                )
            elif score_count % 10 == 0:
                avg = sum(self.state.recent_scores) / len(self.state.recent_scores)
                logger.info(
                    f"📊 [{self.stream.config.name}] "
                    f"raw={raw_violence_score:.1%} stab={stabilized_score:.1%} avg={avg:.1%} "
                    f"mag={motion_analysis.global_motion_magnitude:.1f} "
                    f"stability={motion_analysis.stability_duration:.1f}s {status_str} "
                    f"({inference_time:.0f}ms)"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
    
    def get_smoothed_score(self) -> float:
        """Get smoothed violence score using moving average."""
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)
    
    async def start(self):
        """Start the inference pipeline."""
        if self.is_running:
            return
        
        self.is_running = True
        self._task = asyncio.create_task(self._inference_loop())
    
    async def stop(self):
        """Stop the inference pipeline."""
        self.is_running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "stream_id": self.stream.config.id,
            "is_running": self.is_running,
            "use_local_model": self.use_local,
            "last_inference_time": self.state.last_inference_time.isoformat() if self.state.last_inference_time else None,
            "recent_scores_count": len(self.state.recent_scores),
            "avg_recent_score": sum(self.state.recent_scores) / len(self.state.recent_scores) if self.state.recent_scores else 0,
            "is_in_event": self.state.is_in_event,
            "consecutive_violent_frames": self.state.consecutive_violent_frames
        }

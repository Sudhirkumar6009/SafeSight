"""
Frame Sampler for AI Pipeline
==============================
Samples frames from MediaMTX RTSP proxy for AI processing.
Completely decoupled from live stream delivery.

Architecture:
    MediaMTX RTSP Proxy --> Frame Sampler --> AI Worker Pool
    
Key Features:
1. Independent of display stream
2. Adaptive sampling rate based on system load
3. Thread-safe frame queue for AI workers
"""

import asyncio
import threading
import time
import queue
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Callable, Tuple
from enum import Enum

import cv2
import numpy as np
from loguru import logger


class SamplerState(Enum):
    """Frame sampler states."""
    STOPPED = "stopped"
    CONNECTING = "connecting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class SampledFrame:
    """Container for a sampled frame with metadata."""
    frame: np.ndarray
    stream_id: str
    timestamp: float
    frame_number: int
    quality_level: int  # 1-5 (1=minimal, 5=full)


@dataclass
class SamplerConfig:
    """Configuration for a frame sampler."""
    stream_id: str
    rtsp_url: str  # MediaMTX RTSP proxy URL
    target_fps: float = 5.0  # Default 5 FPS for AI
    quality_level: int = 3   # 1-5 (adaptive)
    buffer_size: int = 32    # Frames to buffer for model
    resize_width: int = 640
    resize_height: int = 360
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = -1  # Infinite


class FrameSampler:
    """
    Samples frames from RTSP stream for AI processing.
    
    Runs independently of display stream.
    Implements adaptive sampling based on system load.
    """
    
    def __init__(
        self,
        config: SamplerConfig,
        on_frame: Optional[Callable[[SampledFrame], None]] = None
    ):
        self.config = config
        self.on_frame = on_frame
        
        self.state = SamplerState.STOPPED
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        
        # Frame buffer for model input
        self.frame_buffer: deque = deque(maxlen=config.buffer_size)
        self._lock = threading.Lock()
        
        # Stats
        self.frame_count = 0
        self.last_frame_time: Optional[float] = None
        self.actual_fps: float = 0.0
        self.error_message: Optional[str] = None
        self.reconnect_attempts = 0
        
        # Adaptive sampling
        self._sample_interval = 1.0 / config.target_fps
        self._quality_level = config.quality_level
        
    def start(self):
        """Start frame sampling in background thread."""
        if self._running:
            return
            
        self._running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        logger.info(f"Frame sampler started for {self.config.stream_id}")
        
    def stop(self):
        """Stop frame sampling."""
        self._running = False
        
        if self.thread:
            self.thread.join(timeout=5.0)
            self.thread = None
            
        if self.capture:
            self.capture.release()
            self.capture = None
            
        self.state = SamplerState.STOPPED
        logger.info(f"Frame sampler stopped for {self.config.stream_id}")
        
    def _connect(self) -> bool:
        """Connect to RTSP stream."""
        self.state = SamplerState.CONNECTING
        
        try:
            if self.capture:
                self.capture.release()
                
            # Build FFmpeg-optimized capture
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'
                'fflags;nobuffer+discardcorrupt|'
                'flags;low_delay|'
                'analyzeduration;500000|'
                'probesize;500000|'
                'max_delay;0'
            )
            
            self.capture = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.capture.isOpened():
                raise ConnectionError(f"Cannot open RTSP: {self.config.rtsp_url}")
                
            # Test read
            ret, _ = self.capture.read()
            if not ret:
                raise ConnectionError("Cannot read from RTSP stream")
                
            self.state = SamplerState.RUNNING
            self.error_message = None
            self.reconnect_attempts = 0
            logger.info(f"Connected to RTSP: {self.config.stream_id}")
            return True
            
        except Exception as e:
            self.error_message = str(e)
            self.state = SamplerState.ERROR
            logger.error(f"Connection failed for {self.config.stream_id}: {e}")
            return False
            
    def _sample_loop(self):
        """Main sampling loop."""
        while self._running:
            try:
                # Connect if needed
                if not self.capture or not self.capture.isOpened():
                    if not self._connect():
                        self.reconnect_attempts += 1
                        if (self.config.max_reconnect_attempts > 0 and 
                            self.reconnect_attempts >= self.config.max_reconnect_attempts):
                            logger.error(f"Max reconnect attempts for {self.config.stream_id}")
                            break
                        time.sleep(self.config.reconnect_delay)
                        continue
                        
                # Sample at target rate
                loop_start = time.time()
                
                # Grab frame
                ret = self.capture.grab()
                if not ret:
                    self.state = SamplerState.ERROR
                    self.error_message = "Frame grab failed"
                    self.capture.release()
                    self.capture = None
                    continue
                    
                # Only decode at sample rate
                ret, frame = self.capture.retrieve()
                if not ret or frame is None:
                    continue
                    
                # Resize for AI
                if self.config.resize_width and self.config.resize_height:
                    frame = cv2.resize(
                        frame,
                        (self.config.resize_width, self.config.resize_height),
                        interpolation=cv2.INTER_AREA
                    )
                    
                # Create sampled frame
                current_time = time.time()
                self.frame_count += 1
                
                sampled = SampledFrame(
                    frame=frame,
                    stream_id=self.config.stream_id,
                    timestamp=current_time,
                    frame_number=self.frame_count,
                    quality_level=self._quality_level
                )
                
                # Add to buffer
                with self._lock:
                    self.frame_buffer.append(sampled)
                    
                # Calculate actual FPS
                if self.last_frame_time:
                    dt = current_time - self.last_frame_time
                    self.actual_fps = 1.0 / dt if dt > 0 else 0
                self.last_frame_time = current_time
                
                # Callback
                if self.on_frame:
                    try:
                        self.on_frame(sampled)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                        
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = max(0, self._sample_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Sample loop error: {e}")
                self.state = SamplerState.ERROR
                self.error_message = str(e)
                time.sleep(1)
                
    def get_frames_for_inference(self, count: int = 16) -> List[np.ndarray]:
        """Get the last N frames for model inference."""
        with self._lock:
            frames = list(self.frame_buffer)
            if len(frames) < count:
                return [f.frame for f in frames]
            return [f.frame for f in frames[-count:]]
            
    def set_quality_level(self, level: int):
        """Set quality level (1-5) for adaptive control."""
        self._quality_level = max(1, min(5, level))
        
        # Adjust sample rate based on quality
        quality_fps_map = {1: 2.0, 2: 3.0, 3: 5.0, 4: 8.0, 5: 10.0}
        self._sample_interval = 1.0 / quality_fps_map.get(level, 5.0)
        
    def get_status(self) -> Dict:
        """Get sampler status."""
        return {
            "stream_id": self.config.stream_id,
            "state": self.state.value,
            "frame_count": self.frame_count,
            "buffer_size": len(self.frame_buffer),
            "actual_fps": round(self.actual_fps, 1),
            "target_fps": self.config.target_fps,
            "quality_level": self._quality_level,
            "error": self.error_message,
        }


class FrameSamplerPool:
    """
    Manages multiple frame samplers for AI processing.
    
    Provides:
    - Centralized sampler management
    - System-wide load balancing
    - Adaptive quality control
    """
    
    def __init__(self, max_samplers: int = 15):
        self.max_samplers = max_samplers
        self.samplers: Dict[str, FrameSampler] = {}
        self._on_frame_callback: Optional[Callable[[SampledFrame], None]] = None
        
    def add_sampler(
        self,
        stream_id: str,
        rtsp_url: str,
        target_fps: float = 5.0,
        auto_start: bool = True
    ) -> bool:
        """Add a new frame sampler."""
        if len(self.samplers) >= self.max_samplers:
            logger.warning(f"Max samplers ({self.max_samplers}) reached")
            return False
            
        if stream_id in self.samplers:
            logger.warning(f"Sampler {stream_id} already exists")
            return False
            
        config = SamplerConfig(
            stream_id=stream_id,
            rtsp_url=rtsp_url,
            target_fps=target_fps,
        )
        
        sampler = FrameSampler(config, on_frame=self._on_frame_callback)
        self.samplers[stream_id] = sampler
        
        if auto_start:
            sampler.start()
            
        return True
        
    def remove_sampler(self, stream_id: str) -> bool:
        """Remove a frame sampler."""
        if stream_id not in self.samplers:
            return False
            
        self.samplers[stream_id].stop()
        del self.samplers[stream_id]
        return True
        
    def get_sampler(self, stream_id: str) -> Optional[FrameSampler]:
        """Get a specific sampler."""
        return self.samplers.get(stream_id)
        
    def set_frame_callback(self, callback: Callable[[SampledFrame], None]):
        """Set callback for all sampled frames."""
        self._on_frame_callback = callback
        for sampler in self.samplers.values():
            sampler.on_frame = callback
            
    def set_global_quality(self, level: int):
        """Set quality level for all samplers."""
        for sampler in self.samplers.values():
            sampler.set_quality_level(level)
            
    def get_all_status(self) -> List[Dict]:
        """Get status of all samplers."""
        return [s.get_status() for s in self.samplers.values()]
        
    def stop_all(self):
        """Stop all samplers."""
        for sampler in self.samplers.values():
            sampler.stop()
        self.samplers.clear()


# Global instance
_sampler_pool: Optional[FrameSamplerPool] = None


def get_sampler_pool() -> FrameSamplerPool:
    """Get or create the global frame sampler pool."""
    global _sampler_pool
    if _sampler_pool is None:
        _sampler_pool = FrameSamplerPool()
    return _sampler_pool

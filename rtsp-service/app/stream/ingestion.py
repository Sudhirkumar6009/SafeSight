"""
RTSP Live Stream Service - Stream Ingestion
============================================
FFmpeg-based RTSP/RTMP stream ingestion with frame buffering
"""

import asyncio
import threading
import time
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from loguru import logger

from app.config import settings


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    stream_id: int
    
    @property
    def shape(self):
        return self.frame.shape


@dataclass
class StreamConfig:
    """Configuration for a stream - optimized for real-time GPU processing."""
    id: int
    name: str
    url: str
    stream_type: str = "rtsp"  # rtsp, rtmp, webcam, file
    reconnect_delay: float = 3.0  # Faster reconnection
    max_reconnect_attempts: int = -1  # -1 for infinite
    target_fps: int = 30  # Higher FPS for real-time (was 15)
    resize_width: Optional[int] = 640
    resize_height: Optional[int] = 360


class FrameBuffer:
    """Thread-safe circular buffer for frames."""
    
    def __init__(self, max_size: int = 1000):  # ~33s at 30fps for 25-30s clips
        self.buffer: deque[FrameData] = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, frame_data: FrameData):
        """Add a frame to the buffer."""
        with self.lock:
            self.buffer.append(frame_data)
    
    def get_window(self, seconds: float, fps: float = 15) -> List[FrameData]:
        """Get frames from the last N seconds."""
        with self.lock:
            frames_needed = int(seconds * fps)
            return list(self.buffer)[-frames_needed:]
    
    def get_all(self) -> List[FrameData]:
        """Get all frames in the buffer."""
        with self.lock:
            return list(self.buffer)
    
    def get_latest(self) -> Optional[FrameData]:
        """Get the most recent frame."""
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
            return None
    
    def get_sampled(self, sample_rate: int = 8, window_seconds: float = 0) -> List[FrameData]:
        """Get sampled frames at specified FPS.
        
        Args:
            sample_rate: Number of frames to sample.
            window_seconds: If > 0, only sample from the last N seconds
                            of the buffer instead of the entire buffer.
        """
        with self.lock:
            frames = list(self.buffer)
            
            # If window_seconds is specified, only use recent frames
            if window_seconds > 0 and frames:
                # Estimate FPS from buffer
                fps_estimate = 30  # default
                if len(frames) >= 2:
                    dt = (frames[-1].timestamp - frames[0].timestamp).total_seconds()
                    if dt > 0:
                        fps_estimate = len(frames) / dt
                
                max_frames = int(window_seconds * fps_estimate)
                if max_frames < len(frames):
                    frames = frames[-max_frames:]
            
            if len(frames) <= sample_rate:
                return frames
            # Sample evenly distributed frames
            indices = np.linspace(0, len(frames) - 1, sample_rate, dtype=int)
            return [frames[i] for i in indices]
    
    def get_latest_consecutive(self, count: int = 16) -> List[FrameData]:
        """Get the last N consecutive frames from the buffer (no sampling/skipping).
        
        This is the CCTV-style method: returns the most recent `count` frames
        exactly as captured, preserving temporal continuity for the model.
        
        Args:
            count: Number of consecutive frames to return.
            
        Returns:
            List of the last `count` FrameData objects, or fewer if buffer
            doesn't have enough frames yet.
        """
        with self.lock:
            if len(self.buffer) <= count:
                return list(self.buffer)
            return list(self.buffer)[-count:]
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)


class StreamIngestion:
    """
    RTSP/RTMP stream ingestion using OpenCV with FFmpeg backend.
    Supports multiple stream types and automatic reconnection.
    """
    
    def __init__(
        self,
        config: StreamConfig,
        on_frame: Optional[Callable[[FrameData], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_status_change: Optional[Callable[[str], None]] = None
    ):
        self.config = config
        self.on_frame = on_frame
        self.on_error = on_error
        self.on_status_change = on_status_change
        
        self.frame_buffer = FrameBuffer(max_size=settings.frame_buffer_size)
        self.is_running = False
        self.is_connected = False
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        
        self.frame_count = 0
        self.last_frame_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.reconnect_attempts = 0
        
        # Frame rate control
        self.target_frame_time = 1.0 / self.config.target_fps
    
    def _build_stream_url(self) -> str:
        """Build the stream URL with appropriate options."""
        url = self.config.url
        
        # For webcam
        if self.config.stream_type == "webcam":
            try:
                return int(url)  # Webcam index
            except ValueError:
                return url
        
        # For file:// URLs, convert to local path
        if url.startswith("file:///"):
            # Convert file:///C:/path to C:/path
            path = url[8:]  # Remove "file:///"
            # Handle URL encoding
            from urllib.parse import unquote
            path = unquote(path)
            return path
        
        if url.startswith("file://"):
            path = url[7:]
            from urllib.parse import unquote
            path = unquote(path)
            return path
        
        # For RTSP streams, add transport options for better compatibility
        if self.config.stream_type == "rtsp" and "rtsp://" in url.lower():
            # OpenCV can handle RTSP directly with FFmpeg backend
            return url
        
        return url
    
    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """Create VideoCapture with optimal settings and GPU hardware decoding."""
        url = self._build_stream_url()
        
        # Check if hardware decoding is enabled
        use_hw_decode = getattr(settings, 'use_hw_decode', True)
        
        # Use FFmpeg backend for RTSP/RTMP with GPU decoding
        if self.config.stream_type in ["rtsp", "rtmp"]:
            if use_hw_decode:
                # Try NVIDIA CUDA/CUVID hardware decoding first
                # FFmpeg RTSP options for low latency with GPU decoding
                hw_url = self._build_hw_decode_url(url)
                cap = cv2.VideoCapture(hw_url, cv2.CAP_FFMPEG)
                
                if cap.isOpened():
                    logger.info(f"Using hardware-accelerated decoding for {url}")
                else:
                    # Fallback to software decoding
                    logger.warning(f"Hardware decoding failed, using software decode for {url}")
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Set buffer size to reduce latency (1-3 frames for real-time)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set FourCC for hardware decoder hint
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
        elif self.config.stream_type == "webcam":
            cap = cv2.VideoCapture(url, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else cv2.CAP_ANY)
        elif self.config.stream_type == "file":
            # For local files, just use the path directly
            cap = cv2.VideoCapture(url)
            logger.info(f"Opening file: {url}")
        else:
            cap = cv2.VideoCapture(url)
        
        # Set capture properties
        if self.config.resize_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resize_width)
        if self.config.resize_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resize_height)
        
        return cap
    
    def _build_hw_decode_url(self, url: str) -> str:
        """Build URL with FFmpeg hardware decode options for low-latency streaming."""
        import os
        
        # Set FFmpeg environment optimized for mobile hotspot/WiFi networks
        # TCP is more reliable than UDP (no packet loss / reordering)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|'        # TCP for reliable, ordered delivery
            'fflags;nobuffer+discardcorrupt|'  # No input buffering, drop corrupt
            'flags;low_delay|'           # Enable low delay mode
            'analyzeduration;1000000|'   # 1s stream analysis (more time for slow networks)
            'probesize;500000|'          # 500KB probe size
            'max_delay;500000|'          # Allow 500ms delay for network jitter
            'reorder_queue_size;10|'     # Small reorder queue for WiFi packet loss
            'stimeout;60000000|'         # 60s socket timeout
            'timeout;60000000'           # 60s connection timeout
        )
        
        return url
    
    def _read_frames(self):
        """Main frame reading loop (runs in separate thread)."""
        logger.info(f"Starting stream ingestion for {self.config.name} ({self.config.url})")
        
        while self.is_running:
            try:
                # Connect to stream
                self.capture = self._create_capture()
                
                if not self.capture or not self.capture.isOpened():
                    raise ConnectionError(f"Failed to open stream: {self.config.url}")
                
                self.is_connected = True
                self.error_message = None
                self.reconnect_attempts = 0
                
                if self.on_status_change:
                    self.on_status_change("running")  # Use "running" to match frontend status
                
                logger.info(f"Connected to stream: {self.config.name}")
                
                # Drain any stale buffered frames on connect
                for _ in range(5):
                    self.capture.grab()
                
                # Frame reading loop
                while self.is_running and self.is_connected:
                    # grab() + retrieve() is faster than read() and
                    # lets us drain stale frames without decoding them
                    if not self.capture.grab():
                        logger.warning(f"Failed to grab frame from {self.config.name}")
                        self.is_connected = False
                        break
                    
                    ret, frame = self.capture.retrieve()
                    
                    if not ret or frame is None:
                        logger.warning(f"Failed to decode frame from {self.config.name}")
                        self.is_connected = False
                        break
                    
                    if not ret or frame is None:
                        logger.warning(f"Failed to read frame from {self.config.name}")
                        self.is_connected = False
                        break
                    
                    # Resize if needed
                    if self.config.resize_width and self.config.resize_height:
                        frame = cv2.resize(
                            frame,
                            (self.config.resize_width, self.config.resize_height),
                            interpolation=cv2.INTER_AREA
                        )
                    
                    # Create frame data
                    self.frame_count += 1
                    self.last_frame_time = datetime.utcnow()
                    
                    frame_data = FrameData(
                        frame=frame,
                        timestamp=self.last_frame_time,
                        frame_number=self.frame_count,
                        stream_id=self.config.id
                    )
                    
                    # Add to buffer
                    self.frame_buffer.add(frame_data)
                    
                    # Callback
                    if self.on_frame:
                        try:
                            self.on_frame(frame_data)
                        except Exception as e:
                            logger.error(f"Error in frame callback: {e}")
                
            except Exception as e:
                self.error_message = str(e)
                self.is_connected = False
                logger.error(f"Stream error ({self.config.name}): {e}")
                
                if self.on_error:
                    self.on_error(str(e))
                if self.on_status_change:
                    self.on_status_change("error")
            
            finally:
                if self.capture:
                    self.capture.release()
                    self.capture = None
            
            # Reconnection logic
            if self.is_running:
                self.reconnect_attempts += 1
                max_attempts = self.config.max_reconnect_attempts
                
                if max_attempts != -1 and self.reconnect_attempts >= max_attempts:
                    logger.error(f"Max reconnection attempts reached for {self.config.name}")
                    self.is_running = False
                    break
                
                if self.on_status_change:
                    self.on_status_change("reconnecting")
                
                logger.info(f"Reconnecting to {self.config.name} in {self.config.reconnect_delay}s...")
                time.sleep(self.config.reconnect_delay)
        
        logger.info(f"Stream ingestion stopped for {self.config.name}")
        if self.on_status_change:
            self.on_status_change("stopped")  # Use "stopped" to match frontend status
    
    def start(self):
        """Start stream ingestion in a background thread."""
        if self.is_running:
            logger.warning(f"Stream {self.config.name} is already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop stream ingestion."""
        logger.info(f"Stopping stream: {self.config.name}")
        self.is_running = False
        self.is_connected = False
        
        if self.thread:
            self.thread.join(timeout=5.0)
            self.thread = None
        
        if self.capture:
            self.capture.release()
            self.capture = None
    
    def get_frame_window(self, seconds: float) -> List[FrameData]:
        """Get frames from the last N seconds."""
        return self.frame_buffer.get_window(seconds, self.config.target_fps)
    
    def get_sampled_frames(self, count: int = 16, window_seconds: float = 0) -> List[FrameData]:
        """Get sampled frames for inference.
        
        Args:
            count: Number of frames to sample.
            window_seconds: If > 0, sample from last N seconds only.
        """
        return self.frame_buffer.get_sampled(count, window_seconds=window_seconds)
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """Get the most recent frame for preview/snapshot."""
        return self.frame_buffer.get_latest()
    
    def get_consecutive_frames(self, count: int = 16) -> List[FrameData]:
        """Get the last N consecutive frames for CCTV-style inference.
        
        Returns the most recent `count` frames with no sampling or skipping,
        preserving exact temporal order for the model's LSTM layer.
        """
        return self.frame_buffer.get_latest_consecutive(count)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stream status."""
        # Compute status string
        if self.error_message:
            status = "error"
        elif self.is_running and self.is_connected:
            status = "running"
        elif self.is_running and not self.is_connected:
            status = "connecting"
        else:
            status = "stopped"
        
        return {
            "id": self.config.id,
            "name": self.config.name,
            "url": self.config.url,
            "rtsp_url": self.config.url,  # Alias for frontend compatibility
            "type": self.config.stream_type,
            "stream_type": self.config.stream_type,  # Alias
            "status": status,  # String status for frontend
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "frame_count": self.frame_count,
            "buffer_size": len(self.frame_buffer),
            "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
            "last_frame_at": self.last_frame_time.isoformat() if self.last_frame_time else None,  # Alias
            "error_message": self.error_message,
            "reconnect_attempts": self.reconnect_attempts
        }


class ClipRecorder:
    """Records video clips from frame buffer."""
    
    def __init__(self, output_dir: str = "./clips"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_clip(
        self,
        frames: List[FrameData],
        stream_name: str,
        event_id: int,
        fps: float = 15.0,
        suffix: str = ""
    ) -> Optional[str]:
        """Save frames as a browser-compatible H.264 MP4 video clip.
        
        Uses PyAV (FFmpeg) for H.264 encoding so clips play in <video> tags.
        suffix can be '_alert' or '_full'.
        
        The fps parameter is only used as a fallback. The actual FPS is calculated
        from frame timestamps to ensure real-time playback speed.
        """
        if not frames:
            logger.warning("No frames to save")
            return None
        
        try:
            import av as _av
            
            # Calculate actual FPS from frame timestamps for real-time playback
            actual_fps = fps  # fallback
            if len(frames) >= 2:
                first_ts = frames[0].timestamp
                last_ts = frames[-1].timestamp
                elapsed = (last_ts - first_ts).total_seconds()
                if elapsed > 0:
                    actual_fps = len(frames) / elapsed
                    # Clamp to reasonable range
                    actual_fps = max(5.0, min(60.0, actual_fps))
                    logger.info(f"📊 Clip FPS: {actual_fps:.1f} (from {len(frames)} frames over {elapsed:.1f}s)")
            
            # Generate filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_name = stream_name.replace(" ", "_").replace("/", "_")
            filename = f"{safe_name}_{event_id}_{timestamp}{suffix}.mp4"
            filepath = self.output_dir / filename
            
            # Get frame dimensions
            height, width = frames[0].frame.shape[:2]
            
            # Encode with PyAV (H.264 in MP4 container — browser-compatible)
            container = _av.open(str(filepath), mode='w')
            stream = container.add_stream('libx264', rate=int(round(actual_fps)))
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            # Fast encoding preset for real-time use
            stream.options = {
                'preset': 'ultrafast',
                'crf': '23',
                'movflags': '+faststart',  # Web-optimised: moov atom at start
            }
            
            for frame_data in frames:
                # OpenCV frames are BGR, convert to RGB for PyAV
                rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
                video_frame = _av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            
            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
            
            clip_duration = len(frames) / actual_fps
            logger.info(f"✅ Saved H.264 clip: {filepath} ({len(frames)} frames, {clip_duration:.1f}s @ {actual_fps:.0f}fps)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save clip: {e}")
            return None
    
    def save_thumbnail(
        self,
        frame: np.ndarray,
        stream_name: str,
        event_id: int
    ) -> Optional[str]:
        """Save a thumbnail image from a frame."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_name = stream_name.replace(" ", "_").replace("/", "_")
            filename = f"{safe_name}_{event_id}_{timestamp}_thumb.jpg"
            filepath = self.output_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            
            logger.info(f"Saved thumbnail: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save thumbnail: {e}")
            return None

"""
ViolenceSense - FFmpeg Ingestion Module
========================================
High-performance FFmpeg-based stream ingestion with auto-reconnection
and frame rate control.

This module provides:
- FFmpeg subprocess management for RTSP/RTMP streams
- Automatic reconnection on stream failure
- Frame rate control and frame skipping
- Memory-efficient frame buffering
- Health monitoring and error reporting
"""

import asyncio
import subprocess
import threading
import time
import signal
from datetime import datetime
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from enum import Enum
import platform
import shutil

import cv2
import numpy as np
from loguru import logger


class StreamStatus(str, Enum):
    """Stream connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class FramePacket:
    """Container for frame data with metadata."""
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    stream_id: str
    pts: Optional[float] = None  # Presentation timestamp from stream
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.frame.shape
    
    @property
    def width(self) -> int:
        return self.frame.shape[1]
    
    @property
    def height(self) -> int:
        return self.frame.shape[0]


@dataclass
class FFmpegConfig:
    """FFmpeg ingestion configuration."""
    # Stream settings
    stream_id: str
    url: str
    stream_type: str = "rtsp"  # rtsp, rtmp, webcam, file
    
    # Connection settings
    connect_timeout: int = 10  # seconds
    read_timeout: int = 5  # seconds (stale stream detection)
    reconnect_delay: float = 3.0
    max_reconnect_attempts: int = -1  # -1 = infinite
    
    # Frame processing
    target_fps: int = 15
    resize_width: int = 640
    resize_height: int = 360
    pixel_format: str = "bgr24"
    
    # Buffer settings
    buffer_size: int = 150  # frames
    queue_size: int = 30  # pending frames
    
    # FFmpeg options
    rtsp_transport: str = "tcp"  # tcp or udp
    hwaccel: Optional[str] = None  # cuda, vaapi, qsv, dxva2, d3d11va
    
    # Output for clip recording
    clips_dir: str = "./clips"


class RingBuffer:
    """
    Thread-safe ring buffer for frame storage.
    Provides O(1) access to recent frames.
    """
    
    def __init__(self, max_size: int = 150):
        self.max_size = max_size
        self.buffer: deque[FramePacket] = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self._frame_count = 0
    
    def push(self, packet: FramePacket) -> None:
        """Add a frame packet to the buffer."""
        with self.lock:
            self.buffer.append(packet)
            self._frame_count += 1
    
    def get_window(self, seconds: float, fps: float = 15) -> List[FramePacket]:
        """Get frames from the last N seconds."""
        with self.lock:
            frames_needed = int(seconds * fps)
            return list(self.buffer)[-frames_needed:]
    
    def get_all(self) -> List[FramePacket]:
        """Get all frames in buffer."""
        with self.lock:
            return list(self.buffer)
    
    def get_sampled(self, num_frames: int) -> List[FramePacket]:
        """Get evenly sampled frames from buffer."""
        with self.lock:
            frames = list(self.buffer)
            if len(frames) <= num_frames:
                return frames
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            return [frames[i] for i in indices]
    
    def get_latest(self, n: int = 1) -> List[FramePacket]:
        """Get the N most recent frames."""
        with self.lock:
            return list(self.buffer)[-n:]
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)
    
    @property
    def frame_count(self) -> int:
        """Total frames received (including dropped)."""
        with self.lock:
            return self._frame_count


class FFmpegIngestion:
    """
    FFmpeg-based stream ingestion with subprocess management.
    
    Uses FFmpeg to:
    - Pull RTSP/RTMP/file streams
    - Handle unstable connections
    - Resize and convert frames
    - Control frame rate
    """
    
    def __init__(
        self,
        config: FFmpegConfig,
        on_frame: Optional[Callable[[FramePacket], None]] = None,
        on_status_change: Optional[Callable[[StreamStatus, Optional[str]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        self.config = config
        self.on_frame = on_frame
        self.on_status_change = on_status_change
        self.on_error = on_error
        
        # State
        self.status = StreamStatus.DISCONNECTED
        self.is_running = False
        self.process: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        
        # Frame buffer
        self.ring_buffer = RingBuffer(max_size=config.buffer_size)
        
        # Statistics
        self.frame_count = 0
        self.last_frame_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.reconnect_count = 0
        self.start_time: Optional[datetime] = None
        
        # Frame rate control
        self._last_process_time = 0
        self._min_frame_interval = 1.0 / config.target_fps
        
        # Verify FFmpeg is available
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is installed and accessible."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError(
                "FFmpeg not found in PATH. Please install FFmpeg:\n"
                "  Windows: choco install ffmpeg OR download from https://ffmpeg.org\n"
                "  Linux: sudo apt install ffmpeg\n"
                "  macOS: brew install ffmpeg"
            )
        logger.debug(f"FFmpeg found at: {ffmpeg_path}")
    
    def _build_ffmpeg_command(self) -> List[str]:
        """Build FFmpeg command optimized for lowest possible latency."""
        url = self.config.url
        
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        
        # Global low-latency flags (before input)
        cmd.extend([
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-analyzeduration", "500000",   # 0.5s analysis
            "-probesize", "500000",         # 0.5s probe
            "-max_delay", "0",
        ])
        
        # Hardware acceleration (if configured)
        if self.config.hwaccel:
            cmd.extend(["-hwaccel", self.config.hwaccel])
        
        # Input options based on stream type
        if self.config.stream_type == "rtsp":
            cmd.extend([
                "-rtsp_transport", self.config.rtsp_transport,
                "-timeout", str(self.config.connect_timeout * 1000000),  # microseconds
                "-stimeout", str(self.config.read_timeout * 1000000),
                "-reorder_queue_size", "0",  # No reorder queue for minimal latency
            ])
        elif self.config.stream_type == "rtmp":
            cmd.extend([
                "-timeout", str(self.config.connect_timeout),
            ])
        elif self.config.stream_type == "webcam":
            # Platform-specific webcam handling
            if platform.system() == "Windows":
                cmd.extend(["-f", "dshow", "-i", f"video={url}"])
            elif platform.system() == "Darwin":  # macOS
                cmd.extend(["-f", "avfoundation", "-i", url])
            else:  # Linux
                cmd.extend(["-f", "v4l2", "-i", url])
            # Skip the regular -i flag for webcam
            url = None
        
        # Add input URL (unless webcam which handles it above)
        if url:
            cmd.extend(["-i", url])
        
        # Output options — scale + fps filter, raw frames to pipe
        cmd.extend([
            "-an",  # No audio
            "-vf", f"scale={self.config.resize_width}:{self.config.resize_height},fps={self.config.target_fps}",
            "-f", "rawvideo",
            "-pix_fmt", self.config.pixel_format,
            "-vsync", "drop",  # Drop duplicate/stale frames
            "pipe:1"  # Output to stdout
        ])
        
        return cmd
    
    def _update_status(self, status: StreamStatus, message: Optional[str] = None) -> None:
        """Update status and notify callback."""
        self.status = status
        if message:
            self.error_message = message
        if self.on_status_change:
            try:
                self.on_status_change(status, message)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def start(self) -> None:
        """Start the ingestion process."""
        if self.is_running:
            logger.warning(f"Stream {self.config.stream_id} already running")
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        self.reconnect_count = 0
        
        # Start reader thread
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"FFmpegReader-{self.config.stream_id}",
            daemon=True
        )
        self.reader_thread.start()
        
        logger.info(f"Started ingestion for stream: {self.config.stream_id}")
    
    def stop(self) -> None:
        """Stop the ingestion process."""
        self.is_running = False
        self._cleanup_process()
        self._update_status(StreamStatus.STOPPED)
        logger.info(f"Stopped ingestion for stream: {self.config.stream_id}")
    
    def _cleanup_process(self) -> None:
        """Cleanup FFmpeg process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.error(f"Error cleaning up FFmpeg process: {e}")
            finally:
                self.process = None
    
    def _reader_loop(self) -> None:
        """Main reader loop with auto-reconnection."""
        while self.is_running:
            try:
                self._connect_and_read()
            except Exception as e:
                logger.error(f"Stream {self.config.stream_id} error: {e}")
                self._update_status(StreamStatus.ERROR, str(e))
                if self.on_error:
                    self.on_error(str(e))
            
            if not self.is_running:
                break
            
            # Check reconnection limits
            if self.config.max_reconnect_attempts > 0:
                if self.reconnect_count >= self.config.max_reconnect_attempts:
                    logger.error(f"Stream {self.config.stream_id}: Max reconnect attempts reached")
                    self._update_status(StreamStatus.ERROR, "Max reconnect attempts reached")
                    break
            
            # Wait before reconnecting
            self.reconnect_count += 1
            logger.info(f"Stream {self.config.stream_id}: Reconnecting in {self.config.reconnect_delay}s (attempt {self.reconnect_count})")
            self._update_status(StreamStatus.RECONNECTING)
            time.sleep(self.config.reconnect_delay)
    
    def _connect_and_read(self) -> None:
        """Connect to stream and read frames."""
        self._update_status(StreamStatus.CONNECTING)
        
        # Build and start FFmpeg process
        cmd = self._build_ffmpeg_command()
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Large buffer for video data
        )
        
        # Frame dimensions
        width = self.config.resize_width
        height = self.config.resize_height
        frame_size = width * height * 3  # BGR24
        
        self._update_status(StreamStatus.CONNECTED)
        logger.info(f"Stream {self.config.stream_id} connected")
        
        # Reset reconnect count on successful connection
        self.reconnect_count = 0
        
        while self.is_running and self.process:
            # Read one frame worth of bytes
            raw_frame = self.process.stdout.read(frame_size)
            
            if len(raw_frame) != frame_size:
                # Stream ended or error
                stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                if stderr:
                    logger.error(f"FFmpeg error: {stderr[:500]}")
                break
            
            # Frame rate control
            current_time = time.time()
            if current_time - self._last_process_time < self._min_frame_interval:
                continue  # Skip frame
            self._last_process_time = current_time
            
            # Convert to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            
            # Create packet
            self.frame_count += 1
            packet = FramePacket(
                frame=frame.copy(),  # Copy to prevent buffer reuse issues
                timestamp=datetime.utcnow(),
                frame_number=self.frame_count,
                stream_id=self.config.stream_id
            )
            
            # Update stats
            self.last_frame_time = packet.timestamp
            
            # Add to ring buffer
            self.ring_buffer.push(packet)
            
            # Notify callback
            if self.on_frame:
                try:
                    self.on_frame(packet)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")
        
        self._cleanup_process()
        self._update_status(StreamStatus.DISCONNECTED)
    
    def get_frame_window(self, seconds: float) -> List[FramePacket]:
        """Get frames from the last N seconds."""
        return self.ring_buffer.get_window(seconds, self.config.target_fps)
    
    def get_sampled_frames(self, num_frames: int = 8) -> List[FramePacket]:
        """Get evenly sampled frames from buffer."""
        return self.ring_buffer.get_sampled(num_frames)
    
    def get_latest_frame(self) -> Optional[FramePacket]:
        """Get the most recent frame."""
        frames = self.ring_buffer.get_latest(1)
        return frames[0] if frames else None
    
    @property
    def is_connected(self) -> bool:
        """Check if stream is currently connected."""
        return self.status == StreamStatus.CONNECTED
    
    @property
    def stats(self) -> dict:
        """Get ingestion statistics."""
        uptime = None
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "stream_id": self.config.stream_id,
            "status": self.status.value,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "frame_count": self.frame_count,
            "buffer_size": len(self.ring_buffer),
            "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
            "reconnect_count": self.reconnect_count,
            "uptime_seconds": uptime,
            "error_message": self.error_message
        }


class ClipRecorder:
    """
    Records video clips from frame buffer.
    Used to save evidence clips on violence detection.
    """
    
    def __init__(self, clips_dir: str = "./clips"):
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
    
    def save_clip(
        self,
        frames: List[FramePacket],
        stream_id: str,
        event_id: str,
        fps: int = 15
    ) -> Optional[str]:
        """
        Save frames as a video clip.
        
        Args:
            frames: List of frame packets to save
            stream_id: Stream identifier
            event_id: Event identifier
            fps: Output video FPS
            
        Returns:
            Path to saved clip or None on failure
        """
        if not frames:
            logger.warning("No frames to save")
            return None
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{stream_id}_{event_id}_{timestamp}.mp4"
        clip_path = self.clips_dir / filename
        
        try:
            # Get frame dimensions from first frame
            height, width = frames[0].shape[:2]
            
            # Use H.264 codec for browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
            
            # Fallback to other H.264 codecs if avc1 fails
            if not writer.isOpened():
                for codec in ['H264', 'x264', 'mp4v']:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
                    if writer.isOpened():
                        break
            
            for packet in frames:
                writer.write(packet.frame)
            
            writer.release()
            
            logger.info(f"Saved clip: {clip_path} ({len(frames)} frames)")
            return str(clip_path)
            
        except Exception as e:
            logger.error(f"Failed to save clip: {e}")
            return None
    
    def save_thumbnail(
        self,
        frame: np.ndarray,
        stream_id: str,
        event_id: str
    ) -> Optional[str]:
        """Save a frame as thumbnail image."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{stream_id}_{event_id}_{timestamp}_thumb.jpg"
        thumb_path = self.clips_dir / filename
        
        try:
            cv2.imwrite(str(thumb_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            logger.debug(f"Saved thumbnail: {thumb_path}")
            return str(thumb_path)
        except Exception as e:
            logger.error(f"Failed to save thumbnail: {e}")
            return None


# Convenience function for creating ingestion from config dict
def create_ingestion_from_config(config_dict: dict) -> FFmpegIngestion:
    """Create FFmpegIngestion from configuration dictionary."""
    config = FFmpegConfig(**config_dict)
    return FFmpegIngestion(config)


if __name__ == "__main__":
    # Test the ingestion module
    import sys
    
    logging_format = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    logger.remove()
    logger.add(sys.stderr, format=logging_format, level="DEBUG")
    
    def on_frame(packet: FramePacket):
        logger.info(f"Frame {packet.frame_number}: {packet.shape}")
    
    def on_status(status: StreamStatus, message: Optional[str]):
        logger.info(f"Status: {status.value} - {message or ''}")
    
    # Test with webcam
    config = FFmpegConfig(
        stream_id="test",
        url="0",
        stream_type="webcam",
        target_fps=10
    )
    
    ingestion = FFmpegIngestion(config, on_frame=on_frame, on_status_change=on_status)
    
    try:
        ingestion.start()
        time.sleep(30)
    finally:
        ingestion.stop()

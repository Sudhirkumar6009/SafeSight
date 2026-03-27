"""
Optimized RTSP Connection Pool
==============================
Connection pooling and management for 10-15 concurrent camera streams.

Key Optimizations:
1. Connection pooling to reduce handshake overhead
2. Automatic reconnection with exponential backoff
3. Health monitoring and stale connection cleanup
4. Resource limits to prevent system overload
"""

import asyncio
import threading
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Callable, Any
from collections import deque
from enum import Enum

import cv2
import numpy as np
from loguru import logger


class ConnectionState(Enum):
    """RTSP connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STALE = "stale"


@dataclass
class ConnectionStats:
    """Statistics for an RTSP connection."""
    frames_received: int = 0
    bytes_received: int = 0
    errors_count: int = 0
    reconnects_count: int = 0
    last_frame_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    avg_fps: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class RTSPConnection:
    """Represents a single RTSP connection."""
    stream_id: str
    url: str
    capture: Optional[cv2.VideoCapture] = None
    state: ConnectionState = ConnectionState.DISCONNECTED
    stats: ConnectionStats = field(default_factory=ConnectionStats)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    
    # Frame timing for FPS calculation
    _frame_times: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        if self.state != ConnectionState.CONNECTED:
            return False
        if not self.capture or not self.capture.isOpened():
            return False
        # Check for stale connection (no frames in 5 seconds)
        if self.stats.last_frame_time:
            stale_threshold = datetime.utcnow() - timedelta(seconds=5)
            if self.stats.last_frame_time < stale_threshold:
                return False
        return True
        
    def update_frame_time(self):
        """Update frame timing statistics."""
        now = datetime.utcnow()
        self._frame_times.append(now)
        self.stats.last_frame_time = now
        self.stats.frames_received += 1
        self.last_used = now
        
        # Calculate FPS from recent frames
        if len(self._frame_times) >= 2:
            elapsed = (self._frame_times[-1] - self._frame_times[0]).total_seconds()
            if elapsed > 0:
                self.stats.avg_fps = len(self._frame_times) / elapsed


class RTSPConnectionPool:
    """
    Pool of RTSP connections for efficient multi-camera management.
    
    Features:
    - Connection pooling and reuse
    - Automatic health monitoring
    - Reconnection with exponential backoff
    - Resource limits
    """
    
    def __init__(
        self,
        max_connections: int = 20,
        health_check_interval: float = 5.0,
        stale_timeout: float = 30.0,
        max_reconnect_attempts: int = 10
    ):
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        self.stale_timeout = stale_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._connections: Dict[str, RTSPConnection] = {}
        self._lock = threading.Lock()
        self._running = False
        self._health_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_connection_change: Optional[Callable[[str, ConnectionState], None]] = None
        self._on_frame: Optional[Callable[[str, np.ndarray], None]] = None
        
    def start(self):
        """Start the connection pool and health monitoring."""
        if self._running:
            return
            
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self._health_thread.start()
        logger.info(f"RTSP connection pool started (max: {self.max_connections})")
        
    def stop(self):
        """Stop the connection pool and close all connections."""
        self._running = False
        
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
            
        # Close all connections
        with self._lock:
            for conn in self._connections.values():
                self._close_connection(conn)
            self._connections.clear()
            
        logger.info("RTSP connection pool stopped")
        
    def _create_capture(self, url: str) -> cv2.VideoCapture:
        """Create an optimized VideoCapture for low-latency streaming."""
        # Set FFmpeg options optimized for mobile hotspot/slow networks
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|'          # TCP for reliability over WiFi
            'fflags;nobuffer+discardcorrupt|'  # No buffering
            'flags;low_delay|'             # Low delay mode
            'max_delay;500000|'            # Allow 500ms delay for network jitter
            'reorder_queue_size;10|'       # Small reorder queue for WiFi packet loss
            'analyzeduration;1000000|'     # 1s analysis (more time for slow networks)
            'probesize;500000|'            # 500KB probe (more data for detection)
            'flush_packets;1|'             # Flush immediately
            'stimeout;60000000|'           # 60s socket timeout (for slow connections)
            'timeout;60000000'             # 60s connection timeout
        )
        
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # Minimal buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set codec hint
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        return cap
        
    def get_connection(self, stream_id: str, url: str) -> Optional[RTSPConnection]:
        """
        Get or create an RTSP connection.
        
        Args:
            stream_id: Unique identifier for the stream
            url: RTSP URL
            
        Returns:
            RTSPConnection if successful, None if pool is full
        """
        with self._lock:
            # Return existing connection
            if stream_id in self._connections:
                conn = self._connections[stream_id]
                conn.last_used = datetime.utcnow()
                return conn
                
            # Check pool limit
            if len(self._connections) >= self.max_connections:
                # Try to evict stale connections
                self._evict_stale_connections()
                if len(self._connections) >= self.max_connections:
                    logger.warning(f"Connection pool full ({self.max_connections})")
                    return None
                    
            # Create new connection
            conn = RTSPConnection(stream_id=stream_id, url=url)
            self._connections[stream_id] = conn
            
        # Connect in background
        threading.Thread(
            target=self._connect,
            args=(conn,),
            daemon=True
        ).start()
        
        return conn
        
    def _connect(self, conn: RTSPConnection):
        """Connect to RTSP stream."""
        conn.state = ConnectionState.CONNECTING
        self._notify_state_change(conn)
        
        try:
            cap = self._create_capture(conn.url)
            
            if not cap.isOpened():
                raise ConnectionError(f"Cannot open RTSP: {conn.url}")
                
            # Flush any stale frames
            for _ in range(3):
                cap.grab()
                
            # Test read
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                raise ConnectionError("Cannot read from RTSP stream")
                
            conn.capture = cap
            conn.state = ConnectionState.CONNECTED
            conn.update_frame_time()
            conn.stats.errors_count = 0
            
            logger.info(f"Connected to RTSP: {conn.stream_id}")
            self._notify_state_change(conn)
            
        except Exception as e:
            conn.state = ConnectionState.ERROR
            conn.stats.last_error_time = datetime.utcnow()
            conn.stats.last_error_message = str(e)
            conn.stats.errors_count += 1
            
            logger.error(f"RTSP connection error ({conn.stream_id}): {e}")
            self._notify_state_change(conn)
            
    def _reconnect(self, conn: RTSPConnection):
        """Attempt to reconnect to RTSP stream."""
        if conn.stats.reconnects_count >= self.max_reconnect_attempts:
            conn.state = ConnectionState.ERROR
            logger.error(f"Max reconnect attempts reached for {conn.stream_id}")
            return
            
        conn.state = ConnectionState.RECONNECTING
        conn.stats.reconnects_count += 1
        self._notify_state_change(conn)
        
        # Exponential backoff
        delay = min(30, 2 ** min(conn.stats.reconnects_count, 5))
        logger.info(f"Reconnecting to {conn.stream_id} in {delay}s (attempt {conn.stats.reconnects_count})")
        time.sleep(delay)
        
        # Close existing capture
        if conn.capture:
            conn.capture.release()
            conn.capture = None
            
        # Reconnect
        self._connect(conn)
        
    def _close_connection(self, conn: RTSPConnection):
        """Close an RTSP connection."""
        if conn.capture:
            conn.capture.release()
            conn.capture = None
        conn.state = ConnectionState.DISCONNECTED
        
    def release_connection(self, stream_id: str):
        """Release a connection back to the pool."""
        with self._lock:
            if stream_id in self._connections:
                conn = self._connections[stream_id]
                self._close_connection(conn)
                del self._connections[stream_id]
                logger.info(f"Released RTSP connection: {stream_id}")
                
    def _evict_stale_connections(self):
        """Evict stale or unused connections."""
        now = datetime.utcnow()
        stale_threshold = now - timedelta(seconds=self.stale_timeout)
        
        to_evict = []
        for stream_id, conn in self._connections.items():
            if conn.last_used < stale_threshold:
                to_evict.append(stream_id)
            elif conn.state == ConnectionState.ERROR:
                to_evict.append(stream_id)
                
        for stream_id in to_evict:
            conn = self._connections.pop(stream_id, None)
            if conn:
                self._close_connection(conn)
                logger.info(f"Evicted stale connection: {stream_id}")
                
    def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while self._running:
            try:
                time.sleep(self.health_check_interval)
                
                with self._lock:
                    connections = list(self._connections.values())
                    
                for conn in connections:
                    if not self._running:
                        break
                        
                    # Check if connection is healthy
                    if conn.state == ConnectionState.CONNECTED:
                        if not conn.is_healthy():
                            conn.state = ConnectionState.STALE
                            logger.warning(f"Connection stale: {conn.stream_id}")
                            threading.Thread(
                                target=self._reconnect,
                                args=(conn,),
                                daemon=True
                            ).start()
                            
                    elif conn.state == ConnectionState.ERROR:
                        # Auto-reconnect on error
                        threading.Thread(
                            target=self._reconnect,
                            args=(conn,),
                            daemon=True
                        ).start()
                        
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                
    def _notify_state_change(self, conn: RTSPConnection):
        """Notify listeners of connection state change."""
        if self._on_connection_change:
            try:
                self._on_connection_change(conn.stream_id, conn.state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
                
    def set_connection_change_callback(self, callback: Callable[[str, ConnectionState], None]):
        """Set callback for connection state changes."""
        self._on_connection_change = callback
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            connections = []
            for conn in self._connections.values():
                connections.append({
                    "stream_id": conn.stream_id,
                    "state": conn.state.value,
                    "frames_received": conn.stats.frames_received,
                    "avg_fps": round(conn.stats.avg_fps, 1),
                    "reconnects": conn.stats.reconnects_count,
                    "errors": conn.stats.errors_count,
                    "healthy": conn.is_healthy(),
                })
                
            return {
                "total_connections": len(self._connections),
                "max_connections": self.max_connections,
                "healthy_connections": sum(1 for c in self._connections.values() if c.is_healthy()),
                "connections": connections,
            }


# Global instance
_connection_pool: Optional[RTSPConnectionPool] = None


def get_connection_pool() -> RTSPConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = RTSPConnectionPool(max_connections=20)
    return _connection_pool


def init_connection_pool():
    """Initialize and start the connection pool."""
    pool = get_connection_pool()
    pool.start()
    return pool

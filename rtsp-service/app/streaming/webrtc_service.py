"""
WebRTC Streaming Service with MediaMTX Integration
===================================================
Ultra-low latency streaming (~200-500ms) for 10-15 cameras.

Architecture:
    Camera RTSP --> MediaMTX --> WebRTC (WHEP) --> Browser
                             --> Frame Sampler --> AI Workers

This module provides:
1. WebRTC WHEP endpoint proxying through MediaMTX
2. Stream management with auto-reconnection
3. Health monitoring for all streams
"""

import asyncio
import aiohttp
import os
import time
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

# MediaMTX Configuration
MEDIAMTX_HOST = os.getenv("MEDIAMTX_HOST", "localhost")
MEDIAMTX_API_PORT = int(os.getenv("MEDIAMTX_API_PORT", "9997"))
MEDIAMTX_RTSP_PORT = int(os.getenv("MEDIAMTX_RTSP_PORT", "8554"))
MEDIAMTX_WEBRTC_PORT = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))
MEDIAMTX_HLS_PORT = int(os.getenv("MEDIAMTX_HLS_PORT", "8888"))


class StreamQuality(Enum):
    """Stream quality presets for adaptive control."""
    ULTRA = "ultra"      # 1080p, 30fps
    HIGH = "high"        # 720p, 30fps
    MEDIUM = "medium"    # 480p, 20fps
    LOW = "low"          # 360p, 15fps
    MINIMAL = "minimal"  # 240p, 10fps


@dataclass
class QualityConfig:
    """Quality configuration for adaptive streaming."""
    width: int
    height: int
    fps: int
    bitrate: str
    ai_sample_rate: int  # How often to sample frames for AI (every N frames)
    
    
QUALITY_PRESETS: Dict[StreamQuality, QualityConfig] = {
    StreamQuality.ULTRA: QualityConfig(1920, 1080, 30, "4M", 3),
    StreamQuality.HIGH: QualityConfig(1280, 720, 30, "2M", 4),
    StreamQuality.MEDIUM: QualityConfig(854, 480, 20, "1M", 5),
    StreamQuality.LOW: QualityConfig(640, 360, 15, "512K", 6),
    StreamQuality.MINIMAL: QualityConfig(426, 240, 10, "256K", 10),
}


@dataclass
class WebRTCStream:
    """WebRTC stream state."""
    stream_id: str
    name: str
    source_rtsp_url: str
    mediamtx_path: str
    quality: StreamQuality = StreamQuality.HIGH
    is_active: bool = False
    is_ready: bool = False
    viewers_count: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    last_health_check: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class WebRTCStreamManager:
    """
    Manages WebRTC streams through MediaMTX.
    
    Provides ultra-low latency delivery while keeping AI processing separate.
    Supports 10-15 simultaneous camera streams.
    """
    
    def __init__(self):
        self.streams: Dict[str, WebRTCStream] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._on_stream_status_change: Optional[Callable] = None
        
        # MediaMTX endpoints
        self.api_url = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_API_PORT}"
        self.rtsp_base = f"rtsp://{MEDIAMTX_HOST}:{MEDIAMTX_RTSP_PORT}"
        self.webrtc_base = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_WEBRTC_PORT}"
        self.hls_base = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_HLS_PORT}"
        
    async def start(self):
        """Start the WebRTC stream manager."""
        if self._running:
            return
            
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        # Check MediaMTX health
        healthy = await self._check_mediamtx_health()
        if not healthy:
            logger.warning("MediaMTX is not responding - streams may not work")
        else:
            logger.info(f"MediaMTX connected at {self.api_url}")
        
        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_monitor_loop())
        
        logger.info("WebRTC Stream Manager started")
        
    async def stop(self):
        """Stop the WebRTC stream manager."""
        self._running = False
        
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        # Remove all streams from MediaMTX
        for stream_id in list(self.streams.keys()):
            await self.remove_stream(stream_id)
        
        if self._session:
            await self._session.close()
            self._session = None
            
        logger.info("WebRTC Stream Manager stopped")
        
    async def add_stream(
        self, 
        stream_id: str, 
        name: str, 
        rtsp_url: str,
        quality: StreamQuality = StreamQuality.HIGH
    ) -> Dict[str, Any]:
        """
        Add a camera stream to MediaMTX for WebRTC delivery.
        
        Args:
            stream_id: Unique stream identifier
            name: Display name
            rtsp_url: Camera RTSP URL
            quality: Initial quality preset
            
        Returns:
            Stream info with WebRTC/RTSP/HLS URLs
        """
        # Generate MediaMTX path name
        path_name = f"cam_{stream_id}"
        
        # Create MediaMTX path configuration
        path_config = {
            "source": rtsp_url,
            "sourceOnDemand": False,  # Always keep connection for low latency
            "sourceOnDemandStartTimeout": "10s",
            "sourceOnDemandCloseAfter": "60s",
            "maxReaders": 100,  # Support many viewers
            "record": False,
        }
        
        # Add path to MediaMTX via API
        try:
            async with self._session.post(
                f"{self.api_url}/v3/config/paths/add/{path_name}",
                json=path_config
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Added stream path {path_name} to MediaMTX")
                elif resp.status == 400:
                    # Path exists, try to update
                    async with self._session.patch(
                        f"{self.api_url}/v3/config/paths/patch/{path_name}",
                        json=path_config
                    ) as patch_resp:
                        if patch_resp.status == 200:
                            logger.info(f"Updated existing path {path_name}")
                        else:
                            error = await patch_resp.text()
                            logger.warning(f"Failed to update path: {error}")
                else:
                    error = await resp.text()
                    logger.error(f"Failed to add path to MediaMTX: {error}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"MediaMTX API error: {e}")
            # Continue - MediaMTX might be configured via file
            
        # Store stream info
        stream = WebRTCStream(
            stream_id=stream_id,
            name=name,
            source_rtsp_url=rtsp_url,
            mediamtx_path=path_name,
            quality=quality,
            is_active=True,
        )
        self.streams[stream_id] = stream
        
        return self.get_stream_urls(stream_id)
        
    async def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream from MediaMTX."""
        if stream_id not in self.streams:
            return False
            
        stream = self.streams[stream_id]
        path_name = stream.mediamtx_path
        
        try:
            async with self._session.delete(
                f"{self.api_url}/v3/config/paths/delete/{path_name}"
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Removed path {path_name} from MediaMTX")
                else:
                    logger.warning(f"Failed to remove path {path_name}")
        except Exception as e:
            logger.error(f"Error removing path: {e}")
            
        del self.streams[stream_id]
        return True
        
    def get_stream_urls(self, stream_id: str) -> Dict[str, Any]:
        """
        Get all playback URLs for a stream.
        
        Returns URLs for:
        - WebRTC (WHEP) - Ultra low latency (~200-500ms)
        - RTSP - For AI frame sampling
        - HLS - Fallback for older browsers
        """
        if stream_id not in self.streams:
            return {}
            
        stream = self.streams[stream_id]
        path = stream.mediamtx_path
        
        return {
            "stream_id": stream_id,
            "name": stream.name,
            "path": path,
            # WebRTC URLs
            "webrtc_url": f"{self.webrtc_base}/{path}",
            "webrtc_whep_url": f"{self.webrtc_base}/{path}/whep",
            # RTSP URL (for AI frame sampling)
            "rtsp_proxy_url": f"{self.rtsp_base}/{path}",
            # HLS fallback
            "hls_url": f"{self.hls_base}/{path}/index.m3u8",
            # Stream info
            "is_ready": stream.is_ready,
            "viewers": stream.viewers_count,
            "quality": stream.quality.value,
        }
        
    async def set_stream_quality(self, stream_id: str, quality: StreamQuality):
        """Change stream quality for adaptive load control."""
        if stream_id not in self.streams:
            return
            
        self.streams[stream_id].quality = quality
        logger.info(f"Stream {stream_id} quality set to {quality.value}")
        
    async def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a stream."""
        if stream_id not in self.streams:
            return None
            
        stream = self.streams[stream_id]
        
        # Fetch live stats from MediaMTX
        try:
            async with self._session.get(
                f"{self.api_url}/v3/paths/get/{stream.mediamtx_path}"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    stream.is_ready = data.get("ready", False)
                    stream.viewers_count = len(data.get("readers", []))
                    stream.bytes_received = data.get("bytesReceived", 0)
                    stream.bytes_sent = data.get("bytesSent", 0)
                    stream.error = None
                else:
                    stream.error = f"MediaMTX API returned {resp.status}"
        except Exception as e:
            stream.error = str(e)
            
        return {
            "stream_id": stream.stream_id,
            "name": stream.name,
            "source_url": stream.source_rtsp_url,
            "mediamtx_path": stream.mediamtx_path,
            "quality": stream.quality.value,
            "is_active": stream.is_active,
            "is_ready": stream.is_ready,
            "viewers_count": stream.viewers_count,
            "bytes_received": stream.bytes_received,
            "bytes_sent": stream.bytes_sent,
            "error": stream.error,
            "urls": self.get_stream_urls(stream_id),
        }
        
    async def list_all_streams(self) -> List[Dict[str, Any]]:
        """List all streams with status."""
        result = []
        for stream_id in self.streams:
            status = await self.get_stream_status(stream_id)
            if status:
                result.append(status)
        return result
        
    async def _check_mediamtx_health(self) -> bool:
        """Check if MediaMTX is responding."""
        try:
            async with self._session.get(f"{self.api_url}/v3/paths/list") as resp:
                return resp.status == 200
        except:
            return False
            
    async def _health_monitor_loop(self):
        """Background task to monitor stream health."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                for stream_id, stream in list(self.streams.items()):
                    try:
                        async with self._session.get(
                            f"{self.api_url}/v3/paths/get/{stream.mediamtx_path}"
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                was_ready = stream.is_ready
                                stream.is_ready = data.get("ready", False)
                                stream.viewers_count = len(data.get("readers", []))
                                stream.last_health_check = datetime.utcnow()
                                stream.error = None
                                
                                # Notify on status change
                                if was_ready != stream.is_ready and self._on_stream_status_change:
                                    await self._on_stream_status_change(stream_id, stream.is_ready)
                            else:
                                stream.is_ready = False
                                stream.error = f"HTTP {resp.status}"
                    except Exception as e:
                        stream.is_ready = False
                        stream.error = str(e)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
                
    def set_status_change_callback(self, callback: Callable):
        """Set callback for stream status changes."""
        self._on_stream_status_change = callback


# Global instance
_webrtc_manager: Optional[WebRTCStreamManager] = None


def get_webrtc_manager() -> WebRTCStreamManager:
    """Get or create the global WebRTC stream manager."""
    global _webrtc_manager
    if _webrtc_manager is None:
        _webrtc_manager = WebRTCStreamManager()
    return _webrtc_manager


async def init_webrtc_manager():
    """Initialize and start the WebRTC manager."""
    manager = get_webrtc_manager()
    await manager.start()
    return manager

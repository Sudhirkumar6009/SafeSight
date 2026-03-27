"""
MediaMTX Manager
================
Manages MediaMTX server for RTSP relay and WebRTC delivery.

MediaMTX acts as the streaming backbone:
- Ingests RTSP streams from cameras
- Serves WebRTC for ultra-low latency browser playback
- Provides RTSP proxy for AI pipeline frame sampling
"""

import asyncio
import aiohttp
import subprocess
import threading
import time
import os
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

# MediaMTX API settings
MEDIAMTX_API_URL = os.getenv("MEDIAMTX_API_URL", "http://localhost:9997")
MEDIAMTX_RTSP_PORT = int(os.getenv("MEDIAMTX_RTSP_PORT", "8554"))
MEDIAMTX_WEBRTC_PORT = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))
MEDIAMTX_HLS_PORT = int(os.getenv("MEDIAMTX_HLS_PORT", "8888"))


@dataclass
class StreamPath:
    """Represents a stream path in MediaMTX."""
    name: str
    source_url: str  # Original camera RTSP URL
    ready: bool = False
    readers: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_check: Optional[datetime] = None
    error: Optional[str] = None


class MediaMTXManager:
    """
    Manages MediaMTX streaming server.
    
    Architecture:
        Camera RTSP → MediaMTX (relay) → WebRTC (browser) + RTSP (AI sampling)
    
    Benefits:
        - Single connection per camera (MediaMTX handles fan-out)
        - WebRTC for ultra-low latency (~200-500ms)
        - RTSP proxy for AI frame sampling (separate from display)
        - HLS fallback for older browsers
    """
    
    def __init__(self):
        self.api_url = MEDIAMTX_API_URL
        self.rtsp_port = MEDIAMTX_RTSP_PORT
        self.webrtc_port = MEDIAMTX_WEBRTC_PORT
        self.hls_port = MEDIAMTX_HLS_PORT
        
        self.streams: Dict[str, StreamPath] = {}
        self._lock = threading.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the MediaMTX manager."""
        if self._running:
            return
            
        self._running = True
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        
        # Start health monitoring
        self._monitor_task = asyncio.create_task(self._health_monitor())
        
        logger.info(f"✅ MediaMTX Manager started (API: {self.api_url})")
        
    async def stop(self):
        """Stop the MediaMTX manager."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
        if self._session:
            await self._session.close()
            self._session = None
            
        logger.info("🛑 MediaMTX Manager stopped")
        
    async def add_stream(self, stream_id: str, rtsp_url: str, name: str = "") -> Dict[str, Any]:
        """
        Add a camera stream to MediaMTX.
        
        Args:
            stream_id: Unique stream identifier
            rtsp_url: Camera's RTSP URL
            name: Optional friendly name
            
        Returns:
            Stream info with WebRTC/RTSP URLs
        """
        path_name = f"cam_{stream_id}"
        
        # Configure the path in MediaMTX to pull from camera
        config = {
            "source": rtsp_url,
            "sourceOnDemand": False,  # Always connected for low latency
            "sourceOnDemandStartTimeout": "10s",
            "sourceOnDemandCloseAfter": "60s",
            "maxReaders": 100,  # Support many viewers
            "readUser": "",
            "readPass": "",
            "publishUser": "",
            "publishPass": "",
            "runOnInit": "",
            "runOnDemand": "",
        }
        
        try:
            async with self._session.post(
                f"{self.api_url}/v3/config/paths/add/{path_name}",
                json=config
            ) as resp:
                if resp.status == 200:
                    logger.info(f"✅ Added stream {path_name} from {rtsp_url}")
                elif resp.status == 400:
                    # Path may already exist, try to patch it
                    async with self._session.patch(
                        f"{self.api_url}/v3/config/paths/patch/{path_name}",
                        json=config
                    ) as patch_resp:
                        if patch_resp.status != 200:
                            logger.warning(f"Failed to update path {path_name}")
                else:
                    error = await resp.text()
                    logger.error(f"Failed to add stream {path_name}: {error}")
                    
        except Exception as e:
            logger.error(f"MediaMTX API error: {e}")
            # Continue anyway - MediaMTX might be configured via file
            
        # Store stream info
        with self._lock:
            self.streams[stream_id] = StreamPath(
                name=name or path_name,
                source_url=rtsp_url,
            )
            
        return self.get_stream_urls(stream_id)
        
    async def remove_stream(self, stream_id: str):
        """Remove a stream from MediaMTX."""
        path_name = f"cam_{stream_id}"
        
        try:
            async with self._session.delete(
                f"{self.api_url}/v3/config/paths/delete/{path_name}"
            ) as resp:
                if resp.status == 200:
                    logger.info(f"🗑️ Removed stream {path_name}")
        except Exception as e:
            logger.warning(f"Failed to remove stream {path_name}: {e}")
            
        with self._lock:
            self.streams.pop(stream_id, None)
            
    def get_stream_urls(self, stream_id: str) -> Dict[str, str]:
        """
        Get all available URLs for a stream.
        
        Returns:
            Dict with webrtc, rtsp, hls URLs
        """
        path_name = f"cam_{stream_id}"
        host = os.getenv("MEDIAMTX_HOST", "localhost")
        
        return {
            "stream_id": stream_id,
            "path": path_name,
            # WebRTC - Ultra low latency (~200-500ms)
            "webrtc": f"http://{host}:{self.webrtc_port}/{path_name}",
            "webrtc_whep": f"http://{host}:{self.webrtc_port}/{path_name}/whep",
            # RTSP - For AI frame sampling
            "rtsp": f"rtsp://{host}:{self.rtsp_port}/{path_name}",
            # HLS - Fallback for older browsers
            "hls": f"http://{host}:{self.hls_port}/{path_name}/index.m3u8",
        }
        
    async def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific stream."""
        path_name = f"cam_{stream_id}"
        
        try:
            async with self._session.get(
                f"{self.api_url}/v3/paths/get/{path_name}"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "stream_id": stream_id,
                        "ready": data.get("ready", False),
                        "readers": len(data.get("readers", [])),
                        "source_ready": data.get("sourceReady", False),
                        "bytes_received": data.get("bytesReceived", 0),
                        "bytes_sent": data.get("bytesSent", 0),
                    }
        except Exception as e:
            logger.debug(f"Failed to get status for {stream_id}: {e}")
            
        return None
        
    async def list_streams(self) -> List[Dict[str, Any]]:
        """List all active streams in MediaMTX."""
        try:
            async with self._session.get(f"{self.api_url}/v3/paths/list") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    streams = []
                    for item in data.get("items", []):
                        name = item.get("name", "")
                        if name.startswith("cam_"):
                            stream_id = name.replace("cam_", "")
                            streams.append({
                                "stream_id": stream_id,
                                "path": name,
                                "ready": item.get("ready", False),
                                "source_ready": item.get("sourceReady", False),
                                "readers": len(item.get("readers", [])),
                                "urls": self.get_stream_urls(stream_id),
                            })
                    return streams
        except Exception as e:
            logger.warning(f"Failed to list streams: {e}")
            
        return []
        
    async def _health_monitor(self):
        """Background task to monitor stream health."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                streams = await self.list_streams()
                for stream in streams:
                    stream_id = stream["stream_id"]
                    ready = stream.get("ready", False)
                    
                    with self._lock:
                        if stream_id in self.streams:
                            self.streams[stream_id].ready = ready
                            self.streams[stream_id].readers = stream.get("readers", 0)
                            self.streams[stream_id].last_check = datetime.utcnow()
                            
                    if not ready:
                        logger.warning(f"⚠️ Stream {stream_id} not ready")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
                
    async def is_healthy(self) -> bool:
        """Check if MediaMTX is responding."""
        try:
            async with self._session.get(f"{self.api_url}/v3/paths/list") as resp:
                return resp.status == 200
        except:
            return False


# Global instance
_mediamtx_manager: Optional[MediaMTXManager] = None


def get_mediamtx_manager() -> MediaMTXManager:
    """Get or create the global MediaMTX manager."""
    global _mediamtx_manager
    if _mediamtx_manager is None:
        _mediamtx_manager = MediaMTXManager()
    return _mediamtx_manager

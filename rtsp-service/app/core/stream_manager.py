"""
ViolenceSense - Production Stream Manager
==========================================
Unified manager for streams, inference pipelines, and event detection.

This manager orchestrates:
- Multiple stream ingestion instances (FFmpeg-based)
- Sliding window inference pipelines  
- Event detection engines
- WebSocket broadcasting for real-time alerts
- Database synchronization
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
import uuid

from loguru import logger
import aiohttp

from app.config import settings
from app.db import (
    init_db, close_db,
    StreamRepository, EventRepository, InferenceLogRepository,
    Stream, StreamStatus as DBStreamStatus
)
from app.stream.ffmpeg_ingestion import (
    FFmpegIngestion, FFmpegConfig, FramePacket, StreamStatus
)
from app.detection.event_engine import (
    EventDetectionEngine, InferenceScore, event_manager
)


@dataclass
class ManagedStream:
    """Container for a fully managed stream with all components."""
    stream_id: str
    name: str
    url: str
    stream_type: str
    location: Optional[str]
    
    # Components
    ingestion: FFmpegIngestion
    detector: EventDetectionEngine
    
    # State
    is_running: bool = False
    inference_task: Optional[asyncio.Task] = None
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "name": self.name,
            "is_running": self.is_running,
            "ingestion": self.ingestion.stats,
            "detector": self.detector.stats
        }


class InferencePipeline:
    """
    Sliding window inference pipeline.
    Samples frames from buffer and sends to ML service for prediction.
    """
    
    def __init__(
        self,
        ingestion: FFmpegIngestion,
        on_result: Callable[[InferenceScore], Any],
        ml_service_url: str = None,
        inference_interval_ms: int = None,
        sample_frames: int = 8
    ):
        self.ingestion = ingestion
        self.on_result = on_result
        self.ml_service_url = ml_service_url or settings.ml_service_url
        self.inference_interval = (inference_interval_ms or settings.inference_interval_ms) / 1000
        self.sample_frames = sample_frames
        
        self._is_running = False
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        """Start the inference loop."""
        self._is_running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.ml_service_timeout)
        )
        
        logger.info(f"Inference pipeline started (interval: {self.inference_interval}s)")
        
        while self._is_running:
            try:
                await self._run_inference()
            except Exception as e:
                logger.error(f"Inference error: {e}")
            
            await asyncio.sleep(self.inference_interval)
    
    async def stop(self):
        """Stop the inference loop."""
        self._is_running = False
        if self._session:
            await self._session.close()
    
    async def _run_inference(self):
        """Run one inference cycle."""
        # Check if we have enough frames
        if len(self.ingestion.ring_buffer) < self.sample_frames:
            return
        
        # Get sampled frames
        frame_packets = self.ingestion.get_sampled_frames(self.sample_frames)
        if not frame_packets:
            return
        
        # Create temporary video from frames for ML service
        import cv2
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"inference_{uuid.uuid4().hex[:8]}.mp4")
        
        try:
            # Write frames to temp video
            height, width = frame_packets[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(temp_path, fourcc, 15, (width, height))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(temp_path, fourcc, 15, (width, height))
            
            for packet in frame_packets:
                writer.write(packet.frame)
            writer.release()
            
            # Send to ML service
            window_start = frame_packets[0].timestamp
            window_end = frame_packets[-1].timestamp
            start_time = datetime.utcnow()
            
            async with self._session.post(
                f"{self.ml_service_url}/api/v1/inference/predict",
                data={"file": open(temp_path, "rb")},
                headers={"Accept": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    # Parse result
                    probs = result.get("probabilities", {})
                    violence_score = probs.get("violence", 0)
                    non_violence_score = probs.get("nonViolence", 1)
                    
                    # Create inference score
                    score = InferenceScore(
                        violence_score=violence_score,
                        non_violence_score=non_violence_score,
                        timestamp=datetime.utcnow(),
                        inference_time_ms=inference_time,
                        frame_count=len(frame_packets),
                        window_start=window_start,
                        window_end=window_end
                    )
                    
                    # Notify callback
                    await self.on_result(score)
                else:
                    logger.warning(f"ML service returned {response.status}")
        
        except aiohttp.ClientError as e:
            logger.warning(f"ML service connection error: {e}")
        except Exception as e:
            logger.error(f"Inference processing error: {e}")
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass


class ProductionStreamManager:
    """
    Production-ready stream manager.
    Manages the complete lifecycle of streams, inference, and event detection.
    """
    
    def __init__(self):
        self.streams: Dict[str, ManagedStream] = {}
        self._websocket_broadcast: Optional[Callable[[str, Dict], Any]] = None
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize the stream manager."""
        if self._is_initialized:
            return
        
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Load existing streams from database
        await self._load_streams_from_db()
        
        self._is_initialized = True
        logger.info("Production stream manager initialized")
    
    async def shutdown(self):
        """Shutdown all streams and cleanup."""
        logger.info("Shutting down stream manager...")
        
        # Stop all streams
        for stream_id in list(self.streams.keys()):
            await self.stop_stream(stream_id)
        
        # Close database
        await close_db()
        
        logger.info("Stream manager shutdown complete")
    
    async def _load_streams_from_db(self):
        """Load active streams from database."""
        try:
            streams = await StreamRepository.get_all_active()
            logger.info(f"Found {len(streams)} active streams in database")
            
            for stream in streams:
                # Register but don't auto-start
                logger.debug(f"Loaded stream: {stream.name} ({stream.id})")
        except Exception as e:
            logger.error(f"Failed to load streams from database: {e}")
    
    def set_broadcast_callback(self, callback: Callable[[str, Dict], Any]):
        """Set WebSocket broadcast callback for real-time updates."""
        self._websocket_broadcast = callback
    
    async def add_stream(
        self,
        name: str,
        url: str,
        stream_type: str = "rtsp",
        location: Optional[str] = None,
        auto_start: bool = False,
        custom_threshold: Optional[float] = None,
        custom_window_seconds: Optional[int] = None
    ) -> str:
        """
        Add a new stream.
        
        Returns the stream ID.
        """
        # Create in database
        stream = await StreamRepository.create(
            name=name,
            url=url,
            stream_type=stream_type,
            location=location,
            custom_threshold=custom_threshold,
            custom_window_seconds=custom_window_seconds
        )
        stream_id = str(stream.id)
        
        # Create managed stream
        managed = await self._create_managed_stream(
            stream_id=stream_id,
            name=name,
            url=url,
            stream_type=stream_type,
            location=location,
            custom_threshold=custom_threshold
        )
        
        self.streams[stream_id] = managed
        logger.info(f"Added stream: {name} (ID: {stream_id})")
        
        if auto_start:
            await self.start_stream(stream_id)
        
        return stream_id
    
    async def _create_managed_stream(
        self,
        stream_id: str,
        name: str,
        url: str,
        stream_type: str,
        location: Optional[str] = None,
        custom_threshold: Optional[float] = None
    ) -> ManagedStream:
        """Create a managed stream with all components."""
        
        # Create FFmpeg config
        config = FFmpegConfig(
            stream_id=stream_id,
            url=url,
            stream_type=stream_type,
            target_fps=settings.frame_sample_rate,
            buffer_size=settings.frame_buffer_size
        )
        
        # Create ingestion
        ingestion = FFmpegIngestion(
            config=config,
            on_status_change=lambda status, msg: self._on_stream_status(stream_id, status, msg)
        )
        
        # Create event detector
        detector = EventDetectionEngine(
            stream_id=stream_id,
            stream_name=name,
            ingestion=ingestion,
            threshold=custom_threshold,
            on_event_start=self._on_event_start,
            on_event_end=self._on_event_end,
            on_alert=self._on_alert
        )
        
        return ManagedStream(
            stream_id=stream_id,
            name=name,
            url=url,
            stream_type=stream_type,
            location=location,
            ingestion=ingestion,
            detector=detector
        )
    
    async def start_stream(self, stream_id: str):
        """Start a stream and its inference pipeline."""
        if stream_id not in self.streams:
            # Try to load from database
            stream = await StreamRepository.get_by_id(stream_id)
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            managed = await self._create_managed_stream(
                stream_id=stream_id,
                name=stream.name,
                url=stream.url,
                stream_type=stream.stream_type,
                location=stream.location,
                custom_threshold=stream.custom_threshold
            )
            self.streams[stream_id] = managed
        
        managed = self.streams[stream_id]
        
        if managed.is_running:
            logger.warning(f"Stream {stream_id} is already running")
            return
        
        # Start ingestion
        managed.ingestion.start()
        
        # Create inference pipeline
        pipeline = InferencePipeline(
            ingestion=managed.ingestion,
            on_result=managed.detector.process_score
        )
        
        # Start inference task
        managed.inference_task = asyncio.create_task(pipeline.start())
        managed.is_running = True
        
        # Update database
        await StreamRepository.update_status(stream_id, DBStreamStatus.ONLINE.value)
        
        logger.info(f"Started stream: {managed.name}")
        
        # Broadcast update
        if self._websocket_broadcast:
            await self._websocket_broadcast("stream_started", {
                "stream_id": stream_id,
                "name": managed.name
            })
    
    async def stop_stream(self, stream_id: str):
        """Stop a stream."""
        if stream_id not in self.streams:
            return
        
        managed = self.streams[stream_id]
        
        # Force end any active event
        managed.detector.force_end_event()
        
        # Cancel inference task
        if managed.inference_task:
            managed.inference_task.cancel()
            try:
                await managed.inference_task
            except asyncio.CancelledError:
                pass
        
        # Stop ingestion
        managed.ingestion.stop()
        managed.is_running = False
        
        # Update database
        await StreamRepository.update_status(stream_id, DBStreamStatus.OFFLINE.value)
        
        logger.info(f"Stopped stream: {managed.name}")
        
        # Broadcast update
        if self._websocket_broadcast:
            await self._websocket_broadcast("stream_stopped", {
                "stream_id": stream_id,
                "name": managed.name
            })
    
    async def remove_stream(self, stream_id: str):
        """Remove a stream completely."""
        # Stop first
        await self.stop_stream(stream_id)
        
        # Remove from manager
        if stream_id in self.streams:
            del self.streams[stream_id]
        
        # Delete from database
        await StreamRepository.delete(stream_id)
        
        logger.info(f"Removed stream: {stream_id}")
    
    def _on_stream_status(self, stream_id: str, status: StreamStatus, message: Optional[str]):
        """Handle stream status changes."""
        logger.debug(f"Stream {stream_id} status: {status.value}")
        
        # Update database async
        asyncio.create_task(
            StreamRepository.update_status(
                stream_id,
                status.value,
                error_message=message if status == StreamStatus.ERROR else None
            )
        )
        
        # Broadcast
        if self._websocket_broadcast:
            asyncio.create_task(
                self._websocket_broadcast("stream_status", {
                    "stream_id": stream_id,
                    "status": status.value,
                    "message": message
                })
            )
    
    def _on_event_start(self, stream_id: str, event_info: Dict):
        """Handle event start."""
        if self._websocket_broadcast:
            asyncio.create_task(
                self._websocket_broadcast("event_started", event_info)
            )
    
    def _on_event_end(self, stream_id: str, event_info: Dict):
        """Handle event end."""
        if self._websocket_broadcast:
            asyncio.create_task(
                self._websocket_broadcast("event_ended", event_info)
            )
    
    def _on_alert(self, event_info: Dict):
        """Handle new alert."""
        logger.warning(f"🚨 ALERT: {event_info.get('stream_name')} - {event_info.get('severity')}")
        
        if self._websocket_broadcast:
            asyncio.create_task(
                self._websocket_broadcast("alert", event_info)
            )
    
    def get_stream(self, stream_id: str) -> Optional[ManagedStream]:
        """Get a stream by ID."""
        return self.streams.get(stream_id)
    
    def get_all_streams(self) -> List[Dict[str, Any]]:
        """Get all streams with their current status."""
        return [stream.stats for stream in self.streams.values()]
    
    def get_running_streams(self) -> List[str]:
        """Get IDs of running streams."""
        return [sid for sid, stream in self.streams.items() if stream.is_running]
    
    async def get_stream_count(self) -> Dict[str, int]:
        """Get stream counts."""
        return {
            "total": len(self.streams),
            "running": len(self.get_running_streams()),
            "stopped": len(self.streams) - len(self.get_running_streams())
        }


# Global instance
stream_manager = ProductionStreamManager()

__all__ = ["ProductionStreamManager", "ManagedStream", "stream_manager"]

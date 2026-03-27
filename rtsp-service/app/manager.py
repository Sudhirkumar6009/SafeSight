"""
RTSP Live Stream Service - Stream Manager
==========================================
Manages multiple streams, pipelines, and event detection
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Stream, Event, EventStatus, async_session, init_db
from app.stream.ingestion import StreamIngestion, StreamConfig, ClipRecorder, FrameData
from app.inference.pipeline import InferencePipeline, InferenceResult
from app.events.detector import EventDetector


@dataclass
class StreamInstance:
    """Container for a managed stream with all components."""
    config: StreamConfig
    ingestion: StreamIngestion
    pipeline: InferencePipeline
    detector: EventDetector
    
    @property
    def id(self) -> int:
        return self.config.id
    
    @property
    def name(self) -> str:
        return self.config.name


@dataclass
class LazyStreamConfig:
    """Lightweight stream config for lazy loading - no model/pipeline created yet."""
    id: int
    name: str
    url: str
    stream_type: str
    location: Optional[str]
    custom_threshold: Optional[float]
    
    def get_status(self) -> dict:
        """Return status for uninitialized stream."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "rtsp_url": self.url,
            "type": self.stream_type,
            "stream_type": self.stream_type,
            "status": "stopped",
            "is_running": False,
            "is_connected": False,
            "frame_count": 0,
            "buffer_size": 0,
            "last_frame_time": None,
            "last_frame_at": None,
            "error_message": None,
            "reconnect_attempts": 0
        }


class StreamManager:
    """
    Central manager for all streams, inference pipelines, and event detection.
    Provides a unified API for stream management.
    """
    
    def __init__(self):
        self.streams: Dict[int, StreamInstance] = {}  # Active streams with full pipeline
        self.lazy_streams: Dict[int, LazyStreamConfig] = {}  # Inactive streams (lazy loaded)
        self.is_initialized = False
        self._websocket_broadcast = None  # Set by API
        
    async def initialize(self):
        """Initialize the stream manager and database."""
        if self.is_initialized:
            return
        
        # Initialize database
        await init_db()
        
        # Load saved streams from database
        await self._load_streams_from_db()
        
        self.is_initialized = True
        logger.info("Stream manager initialized")
    
    async def _load_streams_from_db(self):
        """Load active streams from database (lazy - no model/pipeline created)."""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(Stream).where(Stream.is_active == True)
                )
                db_streams = result.scalars().all()
                
                for db_stream in db_streams:
                    logger.info(f"Loaded stream from DB: {db_stream.name} (lazy)")
                    # Store as lazy config - pipeline will be created when stream is started
                    lazy_config = LazyStreamConfig(
                        id=db_stream.id,
                        name=db_stream.name,
                        url=db_stream.url,
                        stream_type=db_stream.stream_type or "rtsp",
                        location=db_stream.location,
                        custom_threshold=db_stream.custom_threshold
                    )
                    self.lazy_streams[db_stream.id] = lazy_config
                    
        except Exception as e:
            logger.error(f"Failed to load streams from DB: {e}")
    
    def set_broadcast_callback(self, callback):
        """Set WebSocket broadcast callback for real-time updates."""
        self._websocket_broadcast = callback
    
    async def add_stream(
        self,
        name: str,
        url: str,
        stream_type: str = "rtsp",
        location: Optional[str] = None,
        auto_start: bool = False,
        custom_threshold: Optional[float] = None
    ) -> int:
        """Add a new stream."""
        # Save to database
        async with async_session() as session:
            db_stream = Stream(
                name=name,
                url=url,
                stream_type=stream_type,
                location=location,
                is_active=True,
                custom_threshold=custom_threshold
            )
            session.add(db_stream)
            await session.commit()
            await session.refresh(db_stream)
            stream_id = db_stream.id
        
        # Create config
        config = StreamConfig(
            id=stream_id,
            name=name,
            url=url,
            stream_type=stream_type
        )
        
        # Create stream instance
        instance = self._create_stream_instance(config, custom_threshold)
        self.streams[stream_id] = instance
        
        logger.info(f"Added stream: {name} (ID: {stream_id})")
        
        if auto_start:
            await self.start_stream(stream_id)
        
        return stream_id
    
    def _create_stream_instance(
        self,
        config: StreamConfig,
        custom_threshold: Optional[float] = None
    ) -> StreamInstance:
        """Create a complete stream instance with all components."""
        # Get the current event loop for thread-safe callbacks
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        def on_status_change(status):
            """Thread-safe status change callback."""
            if loop:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._on_stream_status_async(config.id, status))
                )
            else:
                self._on_stream_status(config.id, status)
        
        def on_inference_result(result):
            """Thread-safe inference result callback."""
            if loop:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._on_inference_result(config.id, result))
                )
        
        # Create ingestion
        ingestion = StreamIngestion(
            config=config,
            on_status_change=on_status_change
        )
        
        # Create inference pipeline
        pipeline = InferencePipeline(
            stream=ingestion,
            on_result=on_inference_result,
            use_local_model=True
        )
        
        # Create event detector
        detector = EventDetector(
            stream=ingestion,
            pipeline=pipeline,
            on_event_start=self._on_event_start,
            on_event_end=self._on_event_end,
            on_alert=self._on_alert
        )
        
        if custom_threshold:
            detector.threshold = custom_threshold
        
        return StreamInstance(
            config=config,
            ingestion=ingestion,
            pipeline=pipeline,
            detector=detector
        )
    
    async def start_stream(self, stream_id: int):
        """Start a stream and its inference pipeline."""
        # Check if stream is lazy-loaded and needs full initialization
        if stream_id in self.lazy_streams and stream_id not in self.streams:
            lazy_config = self.lazy_streams[stream_id]
            logger.info(f"Initializing lazy stream: {lazy_config.name}")
            
            # Create full config
            config = StreamConfig(
                id=lazy_config.id,
                name=lazy_config.name,
                url=lazy_config.url,
                stream_type=lazy_config.stream_type
            )
            # Create full stream instance with pipeline
            instance = self._create_stream_instance(config, lazy_config.custom_threshold)
            self.streams[stream_id] = instance
            # Remove from lazy streams
            del self.lazy_streams[stream_id]
        
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        instance = self.streams[stream_id]
        
        # Start ingestion
        instance.ingestion.start()
        
        # Start inference pipeline
        await instance.pipeline.start()
        
        # Update database
        await self._update_stream_status(stream_id, "running")
        
        logger.info(f"Started stream: {instance.name}")
    
    async def stop_stream(self, stream_id: int):
        """Stop a stream and its inference pipeline."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        instance = self.streams[stream_id]
        
        # Stop inference pipeline
        await instance.pipeline.stop()
        
        # Stop ingestion
        instance.ingestion.stop()
        
        # Update database
        await self._update_stream_status(stream_id, "stopped")
        
        logger.info(f"Stopped stream: {instance.name}")
    
    async def remove_stream(self, stream_id: int):
        """Remove a stream completely."""
        if stream_id in self.streams:
            await self.stop_stream(stream_id)
            del self.streams[stream_id]
        
        # Also remove from lazy streams if present
        if stream_id in self.lazy_streams:
            del self.lazy_streams[stream_id]
        
        # Mark as inactive in database
        async with async_session() as session:
            await session.execute(
                update(Stream)
                .where(Stream.id == stream_id)
                .values(is_active=False)
            )
            await session.commit()
        
        logger.info(f"Removed stream: {stream_id}")
    
    async def _update_stream_status(self, stream_id: int, status: str, error: str = None):
        """Update stream status in database."""
        try:
            async with async_session() as session:
                values = {"status": status, "updated_at": datetime.utcnow()}
                if error:
                    values["error_message"] = error
                if status == "running":
                    values["last_frame_at"] = datetime.utcnow()
                
                await session.execute(
                    update(Stream)
                    .where(Stream.id == stream_id)
                    .values(**values)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to update stream status: {e}")
    
    def _on_stream_status(self, stream_id: int, status: str):
        """Handle stream status changes (sync version for fallback)."""
        logger.info(f"Stream {stream_id} status: {status}")
    
    async def _on_stream_status_async(self, stream_id: int, status: str):
        """Handle stream status changes (async version)."""
        logger.info(f"Stream {stream_id} status: {status}")
        await self._update_stream_status(stream_id, status)
        
        # Broadcast to WebSockets
        if self._websocket_broadcast:
            await self._websocket_broadcast({
                "type": "stream_status",
                "data": {
                    "stream_id": stream_id,
                    "status": status
                }
            })
    
    async def _on_inference_result(self, stream_id: int, result: InferenceResult):
        """Handle inference results."""
        if stream_id in self.streams:
            detector = self.streams[stream_id].detector
            await detector.process_result(result)
        
        # Broadcast to WebSockets (type must be "inference_score" to match frontend)
        if self._websocket_broadcast:
            await self._websocket_broadcast({
                "type": "inference_score",
                "data": {
                    "stream_id": str(stream_id),
                    "violence_score": result.violence_score,
                    "non_violence_score": result.non_violence_score,
                    "is_violent": result.is_violent,
                    "timestamp": result.timestamp.isoformat(),
                    # Enhanced inference metrics (from ONNX migration)
                    "inference_time_ms": result.inference_time_ms,
                    "raw_score": result.raw_score,
                    "stabilized_score": result.stabilized_score,
                    "is_confirmed": result.is_confirmed,
                    "is_camera_shake": result.is_camera_shake,
                    "is_stable": result.is_stable,
                }
            })
    
    def _on_event_start(self, stream_id: int, event_data: Dict):
        """Handle event start or violence_alert."""
        msg_type = event_data.pop("type", "event_start")
        logger.warning(f"🚨 {msg_type.upper()}: Stream {stream_id} - {event_data}")
        
        if self._websocket_broadcast:
            asyncio.create_task(self._websocket_broadcast({
                "type": msg_type,
                "data": {
                    "stream_id": str(stream_id),
                    **event_data
                }
            }))
    
    def _on_event_end(self, stream_id: int, event_data: Dict):
        """Handle event end."""
        msg_type = event_data.pop("type", "event_end")
        logger.info(f"✅ {msg_type.upper()}: Stream {stream_id} - {event_data}")
        
        if self._websocket_broadcast:
            asyncio.create_task(self._websocket_broadcast({
                "type": msg_type,
                "data": {
                    "stream_id": str(stream_id),
                    **event_data
                }
            }))
    
    def _on_alert(self, event: Event):
        """Handle new alert."""
        if event:
            logger.warning(f"🔔 NEW ALERT: {event.stream_name} - Severity: {event.severity}")
        else:
            logger.warning("🔔 NEW ALERT triggered")
    
    def get_stream_status(self, stream_id: int) -> Optional[Dict[str, Any]]:
        """Get status for a specific stream."""
        # Check active streams first
        if stream_id in self.streams:
            instance = self.streams[stream_id]
            return {
                "stream": instance.ingestion.get_status(),
                "pipeline": instance.pipeline.get_status(),
                "detector": instance.detector.get_status()
            }
        # Check lazy streams
        if stream_id in self.lazy_streams:
            lazy_config = self.lazy_streams[stream_id]
            return {
                "stream": lazy_config.get_status(),
                "pipeline": {"is_running": False, "model_loaded": False},
                "detector": {"in_event": False}
            }
        return None
    
    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status for all streams."""
        all_ids = list(self.streams.keys()) + list(self.lazy_streams.keys())
        return [self.get_stream_status(sid) for sid in all_ids]
    
    async def get_events(
        self,
        status: Optional[EventStatus] = None,
        stream_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Event]:
        """Get events from database."""
        async with async_session() as session:
            query = select(Event).order_by(Event.created_at.desc())
            
            if status:
                query = query.where(Event.status == status)
            if stream_id:
                query = query.where(Event.stream_id == stream_id)
            
            query = query.limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def update_event_status(
        self,
        event_id: int,
        status: EventStatus,
        reviewed_by: str = None,
        notes: str = None
    ):
        """Update event status (confirm/dismiss)."""
        async with async_session() as session:
            values = {
                "status": status,
                "reviewed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            if reviewed_by:
                values["reviewed_by"] = reviewed_by
            if notes:
                values["notes"] = notes
            
            await session.execute(
                update(Event)
                .where(Event.id == event_id)
                .values(**values)
            )
            await session.commit()
        
        logger.info(f"Event {event_id} updated to {status}")
    
    async def shutdown(self):
        """Stop all streams and cleanup."""
        logger.info("Shutting down stream manager...")
        
        for stream_id in list(self.streams.keys()):
            try:
                await self.stop_stream(stream_id)
            except Exception as e:
                logger.error(f"Error stopping stream {stream_id}: {e}")
        
        self.streams.clear()
        logger.info("Stream manager shutdown complete")


# Global manager instance
stream_manager = StreamManager()

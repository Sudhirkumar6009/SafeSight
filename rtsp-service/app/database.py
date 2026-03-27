"""
RTSP Live Stream Service - Database Models
===========================================
SQLAlchemy models for event storage, video clips, and face extraction
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Enum as SQLEnum, ForeignKey, LargeBinary
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import enum

from app.config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True
)

# Create async session factory
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for models
Base = declarative_base()


class EventStatus(str, enum.Enum):
    """Event status enumeration."""
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"
    AUTO_DISMISSED = "AUTO_DISMISSED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    NO_ACTION_REQUIRED = "NO_ACTION_REQUIRED"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Stream(Base):
    """Registered RTSP streams."""
    __tablename__ = "streams"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    url = Column(String(1024), nullable=False)
    stream_type = Column(String(50), default="rtsp")  # rtsp, rtmp, webcam, file
    location = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Stream status
    status = Column(String(50), default="disconnected")  # connected, disconnected, error
    last_frame_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Configuration overrides
    custom_threshold = Column(Float, nullable=True)
    custom_window_seconds = Column(Integer, nullable=True)


class VideoClip(Base):
    """Video clips stored with metadata. Files stored on user's device."""
    __tablename__ = "video_clips"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False)
    stream_id = Column(Integer, ForeignKey("streams.id", ondelete="SET NULL"), nullable=True)
    
    # File information
    filename = Column(String(512), nullable=False)  # Just the filename
    file_path = Column(String(2048), nullable=False)  # Full absolute path on user's device
    file_size_bytes = Column(Integer, nullable=True)
    
    # Video metadata
    duration_seconds = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    codec = Column(String(50), nullable=True)
    frame_count = Column(Integer, nullable=True)
    
    # Thumbnail stored as binary (small JPEG, typically <100KB)
    thumbnail_data = Column(LargeBinary, nullable=True)
    thumbnail_mime_type = Column(String(50), default="image/jpeg")
    
    # Timestamps
    recorded_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    event = relationship("Event", back_populates="video_clip")
    extracted_faces = relationship("ExtractedFace", back_populates="video_clip", cascade="all, delete-orphan")


class ExtractedFace(Base):
    """Extracted face images stored as binary in PostgreSQL."""
    __tablename__ = "extracted_faces"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    clip_id = Column(Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), nullable=False)
    stream_id = Column(Integer, ForeignKey("streams.id", ondelete="SET NULL"), nullable=True)
    event_id = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=True)
    
    # Face image stored as binary (JPEG, typically 10-50KB each)
    image_data = Column(LargeBinary, nullable=False)
    image_mime_type = Column(String(50), default="image/jpeg")
    image_size_bytes = Column(Integer, nullable=True)
    
    # Face metadata
    face_index = Column(Integer, nullable=False)  # 0, 1, 2... for multiple faces
    confidence = Column(Float, nullable=True)  # Face detection confidence
    
    # Bounding box (normalized 0-1 or pixel coordinates)
    bbox_x = Column(Integer, nullable=True)
    bbox_y = Column(Integer, nullable=True)
    bbox_width = Column(Integer, nullable=True)
    bbox_height = Column(Integer, nullable=True)
    
    # Frame info
    frame_number = Column(Integer, nullable=True)
    frame_timestamp_ms = Column(Integer, nullable=True)  # Milliseconds into the clip
    
    # Timestamps
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video_clip = relationship("VideoClip", back_populates="extracted_faces")


class Event(Base):
    """Violence detection events."""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stream_id = Column(Integer, nullable=False)
    stream_name = Column(String(255), nullable=False)
    
    # Detection details
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Confidence scores
    max_confidence = Column(Float, nullable=False)
    avg_confidence = Column(Float, nullable=False)
    min_confidence = Column(Float, nullable=False)
    frame_count = Column(Integer, default=0)
    
    # Classification
    severity = Column(SQLEnum(AlertSeverity), default=AlertSeverity.MEDIUM)
    status = Column(SQLEnum(EventStatus), default=EventStatus.PENDING)
    
    # Legacy clip information (kept for backward compatibility)
    clip_path = Column(String(1024), nullable=True)
    clip_duration = Column(Float, nullable=True)
    thumbnail_path = Column(String(1024), nullable=True)
    
    # Secure encrypted storage references (new)
    secure_clip_id = Column(String(64), nullable=True)  # Encrypted clip filename
    secure_thumbnail_id = Column(String(64), nullable=True)  # Encrypted thumbnail filename
    
    # Face count from extraction
    person_count = Column(Integer, default=0)
    
    # Review
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    video_clip = relationship("VideoClip", back_populates="event", uselist=False, cascade="all, delete-orphan")


class InferenceLog(Base):
    """Log of all inference results for analytics."""
    __tablename__ = "inference_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stream_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Inference results
    violence_score = Column(Float, nullable=False)
    non_violence_score = Column(Float, nullable=False)
    inference_time_ms = Column(Float, nullable=True)
    
    # Frame info
    frame_number = Column(Integer, nullable=True)
    window_start = Column(DateTime, nullable=True)
    window_end = Column(DateTime, nullable=True)


async def init_db():
    """Initialize the database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get a database session."""
    async with async_session() as session:
        yield session

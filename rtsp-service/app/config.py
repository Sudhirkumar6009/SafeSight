"""
RTSP Live Stream Service - Configuration
=========================================
Centralized configuration using pydantic-settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server Settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # ML Service Configuration
    ml_service_url: str = Field(default="http://localhost:8000", alias="ML_SERVICE_URL")
    ml_service_timeout: int = Field(default=30, alias="ML_SERVICE_TIMEOUT")
    
    # Stream Settings - Optimized for low-latency display + reliable detection
    frame_buffer_size: int = Field(default=1000, alias="FRAME_BUFFER_SIZE")  # ~33s at 30fps for 25-30s clips
    sliding_window_seconds: int = Field(default=2, alias="SLIDING_WINDOW_SECONDS")  # 2s sliding window
    frame_sample_rate: int = Field(default=16, alias="FRAME_SAMPLE_RATE")  # 16 frames for model
    inference_interval_ms: int = Field(default=200, alias="INFERENCE_INTERVAL_MS")  # 5 inferences/sec
    target_fps: int = Field(default=30, alias="TARGET_FPS")  # 30 FPS display capture
    
    # GPU Settings
    use_gpu: bool = Field(default=True, alias="USE_GPU")
    gpu_memory_fraction: float = Field(default=0.7, alias="GPU_MEMORY_FRACTION")  # Use 70% of GPU memory
    use_tensorrt: bool = Field(default=False, alias="USE_TENSORRT")  # TensorRT optimization
    use_hw_decode: bool = Field(default=True, alias="USE_HW_DECODE")  # Hardware video decoding
    
    # Event Detection Thresholds
    violence_threshold: float = Field(default=0.50, alias="VIOLENCE_THRESHOLD")
    clip_confidence_threshold: float = Field(default=0.90, alias="CLIP_CONFIDENCE_THRESHOLD")  # 90%+ triggers clip recording
    min_consecutive_frames: int = Field(default=3, alias="MIN_CONSECUTIVE_FRAMES")  # Need 3+ consecutive high scores
    alert_cooldown_seconds: int = Field(default=5, alias="ALERT_COOLDOWN_SECONDS")
    clip_duration_before: int = Field(default=5, alias="CLIP_DURATION_BEFORE")  # Quick alert clip
    clip_duration_after: int = Field(default=15, alias="CLIP_DURATION_AFTER")  # Quick alert clip
    full_clip_before: int = Field(default=10, alias="FULL_CLIP_BEFORE")  # Full evidence clip
    full_clip_after: int = Field(default=10, alias="FULL_CLIP_AFTER")  # Full evidence clip
    min_event_duration_seconds: float = Field(default=1.0, alias="MIN_EVENT_DURATION_SECONDS")
    
    # Shake Detection Settings
    shake_confirmation_seconds: float = Field(default=4.0, alias="SHAKE_CONFIRMATION_SECONDS")  # Require 4s sustained for confirmation
    shake_score_penalty: float = Field(default=0.6, alias="SHAKE_SCORE_PENALTY")  # Reduce score by 40% during shake
    
    # Storage
    clips_dir: str = Field(default="./clips", alias="CLIPS_DIR")
    clips_retention_days: int = Field(default=7, alias="CLIPS_RETENTION_DAYS")
    
    # Database - PostgreSQL (production) or SQLite (development)
    # PostgreSQL: postgresql://user:password@localhost:5432/safesight
    # SQLite: sqlite+aiosqlite:///./events.db
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/violencesense",
        alias="DATABASE_URL"
    )
    
    # Backend Service (for forwarding events)
    backend_url: str = Field(default="http://localhost:5000", alias="BACKEND_URL")
    
    # Model Path
    model_path: Optional[str] = Field(default="../ml-service/models/violence_model_legacy.h5", alias="MODEL_PATH")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/rtsp-service.log", alias="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def ensure_directories(self):
        """Create required directories if they don't exist."""
        Path(self.clips_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()

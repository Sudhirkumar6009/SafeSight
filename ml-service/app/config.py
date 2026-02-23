import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model settings
    default_model_path: str = "./models/violence_model_legacy.h5"
    model_architecture: str = "keras-cnn"
    
    # Inference settings
    num_frames: int = 16  # MobileNetV2-LSTM model expects 16 frames
    frame_size: int = 224
    batch_size: int = 1
    
    # Device settings
    device: str = "cpu"
    use_fp16: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

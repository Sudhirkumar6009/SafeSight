import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model settings
    default_model_path: str = "./models/violence_model.onnx"
    model_architecture: str = "onnx"
    
    # Inference settings
    num_frames: int = 16  # MobileNetV2-LSTM model expects 16 frames
    frame_size: int = 224
    batch_size: int = 1
    
    # Model output interpretation
    # If True: HIGH sigmoid output = VIOLENCE (default, standard training)
    # If False: HIGH sigmoid output = NON-VIOLENCE (inverted labels)
    violence_label_high: bool = True
    
    # Device settings
    device: str = "cpu"
    use_fp16: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

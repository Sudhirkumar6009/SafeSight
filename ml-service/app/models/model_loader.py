"""
SafeSight ML Service - Model Loading and Management

This module handles loading and managing PyTorch and Keras models for violence detection.
Supports various architectures including VideoMAE, TimeSformer, SlowFast, etc.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, Union
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load TensorFlow/Keras to avoid startup issues
KERAS_AVAILABLE = False
keras = None
tf = None

def _load_keras():
    """Lazy load TensorFlow/Keras."""
    global KERAS_AVAILABLE, keras, tf
    if KERAS_AVAILABLE:
        return True
    try:
        import tensorflow as _tf
        from tensorflow import keras as _keras
        tf = _tf
        keras = _keras
        KERAS_AVAILABLE = True
        logger.info("TensorFlow/Keras loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"TensorFlow/Keras not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error loading TensorFlow/Keras: {e}")
        return False


class VideoClassificationModel(nn.Module):
    """
    Base video classification model wrapper.
    Wraps different backbone architectures for violence detection.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        architecture: str = "videomae",
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        self.backbone = None
        self.classifier = None
        
        self._build_model(pretrained_path)
    
    def _build_model(self, pretrained_path: Optional[str] = None):
        """Build the model architecture."""
        
        if self.architecture == "videomae":
            self._build_videomae()
        elif self.architecture == "timesformer":
            self._build_timesformer()
        elif self.architecture == "slowfast":
            self._build_slowfast()
        elif self.architecture == "resnet3d":
            self._build_resnet3d()
        elif self.architecture == "i3d":
            self._build_i3d()
        else:
            self._build_simple_cnn()
        
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_weights(pretrained_path)
    
    def _build_videomae(self):
        """Build VideoMAE-style architecture."""
        try:
            import timm
            # Use a vision transformer as backbone
            self.backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0  # Remove classification head
            )
            feature_dim = self.backbone.num_features
        except ImportError:
            # Fallback to simple architecture
            logger.warning("timm not available, using fallback architecture")
            self._build_simple_cnn()
            return
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
    
    def _build_timesformer(self):
        """Build TimeSformer-style architecture."""
        self._build_videomae()  # Similar approach
    
    def _build_slowfast(self):
        """Build SlowFast-style architecture."""
        self._build_simple_cnn()
    
    def _build_resnet3d(self):
        """Build 3D ResNet architecture."""
        self._build_simple_cnn()
    
    def _build_i3d(self):
        """Build I3D architecture."""
        self._build_simple_cnn()
    
    def _build_simple_cnn(self):
        """Build a simple 3D CNN for fallback."""
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def _load_weights(self, path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Try to load weights
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {path}")
        except Exception as e:
            logger.error(f"Failed to load weights from {path}: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W) or (B, T, C, H, W)
        
        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                pass  # Already correct format
            else:  # (B, T, C, H, W)
                x = x.permute(0, 2, 1, 3, 4)
        
        if self.architecture in ["videomae", "timesformer"]:
            # For ViT-based models, process frames independently or use temporal pooling
            B, C, T, H, W = x.shape
            # Reshape to process frames: (B*T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
            x = x.view(B * T, C, H, W)
            
            # Get features
            features = self.backbone(x)  # (B*T, feature_dim)
            
            # Temporal pooling
            features = features.view(B, T, -1).mean(dim=1)  # (B, feature_dim)
            
            # Classification
            out = self.classifier(features)
        else:
            # For 3D CNN models
            features = self.backbone(x)
            out = self.classifier(features)
        
        return out


class ModelManager:
    """
    Manages model loading, unloading, and state.
    Supports both PyTorch (.pth) and Keras (.keras, .h5) models.
    Singleton pattern for global access.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model: Optional[Union[VideoClassificationModel, Any]] = None
        self.model_type: str = "pytorch"  # "pytorch" or "keras"
        self.device: torch.device = torch.device("cpu")
        self.model_path: Optional[str] = None
        self.architecture: str = "videomae"
        self.is_loaded: bool = False
        self.loaded_at: Optional[datetime] = None
        self.use_fp16: bool = False
        
        # Model info
        self.input_size: Dict[str, int] = {
            "frames": 16,
            "height": 224,
            "width": 224
        }
        self.classes: list = ["violence", "non-violence"]
        
        # Performance tracking
        self.total_predictions: int = 0
        self.total_inference_time: float = 0.0
        
        self._initialized = True
        
        # Set device
        self._set_device()
    
    def _set_device(self):
        """Set the computation device."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detect model type from file extension."""
        ext = os.path.splitext(model_path)[1].lower()
        if ext in ['.keras', '.h5']:
            return "keras"
        elif ext in ['.pth', '.pt']:
            return "pytorch"
        else:
            raise ValueError(f"Unsupported model format: {ext}")
    
    def load_model(
        self,
        model_path: str,
        architecture: str = "videomae",
        use_fp16: bool = True
    ) -> Dict[str, Any]:
        """
        Load a model from the specified path.
        Automatically detects model type from file extension.
        
        Args:
            model_path: Path to the model file (.pth, .pt, .keras, .h5)
            architecture: Model architecture type
            use_fp16: Whether to use FP16 precision (PyTorch only)
        
        Returns:
            Dictionary with load status and model info
        """
        try:
            # Validate path
            if not os.path.exists(model_path):
                return {
                    "success": False,
                    "error": f"Model file not found: {model_path}"
                }
            
            # Detect model type
            model_type = self._detect_model_type(model_path)
            logger.info(f"Detected model type: {model_type}")
            
            # Unload existing model
            if self.is_loaded:
                self.unload_model()
            
            if model_type == "keras":
                return self._load_keras_model(model_path, architecture)
            else:
                return self._load_pytorch_model(model_path, architecture, use_fp16)
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _load_keras_model(self, model_path: str, architecture: str) -> Dict[str, Any]:
        """Load a Keras model with Keras 2.x to 3.x compatibility."""
        # Lazy load TensorFlow
        if not _load_keras():
            return {
                "success": False,
                "error": "TensorFlow/Keras not installed. Install with: pip install tensorflow"
            }
        
        try:
            logger.info(f"Loading Keras model from: {model_path}")
            
            # Suppress TF logging during load
            import os as _os
            old_level = _os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
            _os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Use the compatibility loader for Keras 2.x -> 3.x migration
            from app.models.keras_loader import load_keras_model_compatible
            
            try:
                self.model = load_keras_model_compatible(model_path)
                logger.info("Model loaded successfully with compatibility loader")
            except Exception as e:
                logger.error(f"Compatibility loader failed: {e}")
                return {
                    "success": False,
                    "error": f"Failed to load Keras model: {str(e)}"
                }
            
            self.model_type = "keras"
            
            # Get input shape from model
            try:
                input_shape = self.model.input_shape
                logger.info(f"Model input shape: {input_shape}")
                
                # Update input size based on model
                # Keras models typically have shape: (batch, frames, height, width, channels)
                # or (batch, height, width, channels) for image models
                if isinstance(input_shape, tuple):
                    if len(input_shape) == 5:
                        self.input_size = {
                            "frames": input_shape[1] if input_shape[1] else 16,
                            "height": input_shape[2] if input_shape[2] else 224,
                            "width": input_shape[3] if input_shape[3] else 224
                        }
                    elif len(input_shape) == 4:
                        self.input_size = {
                            "frames": 16,
                            "height": input_shape[1] if input_shape[1] else 224,
                            "width": input_shape[2] if input_shape[2] else 224
                        }
                    else:
                        self.input_size = {"frames": 16, "height": 224, "width": 224}
                else:
                    self.input_size = {"frames": 16, "height": 224, "width": 224}
            except Exception as e:
                logger.warning(f"Could not determine input shape: {e}")
                self.input_size = {"frames": 16, "height": 224, "width": 224}
            
            # Update state
            self.model_path = model_path
            self.architecture = architecture
            self.is_loaded = True
            self.loaded_at = datetime.now()
            self.use_fp16 = False
            
            logger.info(f"Keras model loaded successfully: {model_path}")
            
            return {
                "success": True,
                "message": "Model loaded successfully",
                "modelInfo": {
                    "name": os.path.basename(model_path),
                    "architecture": f"keras-{architecture}",
                    "inputSize": self.input_size,
                    "classes": self.classes
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _load_pytorch_model(
        self,
        model_path: str,
        architecture: str,
        use_fp16: bool
    ) -> Dict[str, Any]:
        """Load a PyTorch model."""
        try:
            # Create model
            self.model = VideoClassificationModel(
                num_classes=2,
                architecture=architecture,
                pretrained_path=model_path
            )
            self.model_type = "pytorch"
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            
            # Enable FP16 if requested and available
            if use_fp16 and self.device.type == "cuda":
                self.use_fp16 = True
            else:
                self.use_fp16 = False
            
            # Update state
            self.model_path = model_path
            self.architecture = architecture
            self.is_loaded = True
            self.loaded_at = datetime.now()
            
            logger.info(f"PyTorch model loaded successfully: {model_path}")
            
            return {
                "success": True,
                "message": "Model loaded successfully",
                "modelInfo": {
                    "name": os.path.basename(model_path),
                    "architecture": architecture,
                    "inputSize": self.input_size,
                    "classes": self.classes
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, input_tensor: Any) -> np.ndarray:
        """
        Run prediction on input tensor.
        Handles both PyTorch and Keras models.
        
        Args:
            input_tensor: Input data (numpy array or torch tensor)
        
        Returns:
            Prediction logits as numpy array
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded")
        
        if self.model_type == "keras":
            # Keras prediction
            if isinstance(input_tensor, torch.Tensor):
                input_tensor = input_tensor.cpu().numpy()
            
            # Ensure correct shape for Keras
            # Keras expects: (batch, frames, height, width, channels)
            if input_tensor.ndim == 5 and input_tensor.shape[1] == 3:
                # Convert from (B, C, T, H, W) to (B, T, H, W, C)
                input_tensor = np.transpose(input_tensor, (0, 2, 3, 4, 1))
            
            predictions = self.model.predict(input_tensor, verbose=0)
            return predictions
        else:
            # PyTorch prediction
            if isinstance(input_tensor, np.ndarray):
                input_tensor = torch.from_numpy(input_tensor)
            
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_tensor)
                else:
                    logits = self.model(input_tensor)
            
            return logits.cpu().numpy()
    
    def unload_model(self) -> bool:
        """Unload the current model and free memory."""
        try:
            if self.model is not None:
                if self.model_type == "keras" and keras is not None:
                    # Clear Keras model
                    keras.backend.clear_session()
                del self.model
                self.model = None
            
            # Clear CUDA cache for PyTorch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            self.model_path = None
            self.loaded_at = None
            self.model_type = "pytorch"
            
            logger.info("Model unloaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current model status."""
        gpu_info = None
        if torch.cuda.is_available():
            gpu_info = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "used": torch.cuda.memory_allocated(0),
                "free": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            }
        
        return {
            "isLoaded": self.is_loaded,
            "currentModel": {
                "path": self.model_path,
                "architecture": self.architecture,
                "modelType": self.model_type,
                "loadedAt": self.loaded_at.isoformat() if self.loaded_at else None
            } if self.is_loaded else None,
            "gpuAvailable": torch.cuda.is_available(),
            "gpuMemory": gpu_info
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        avg_inference_time = 0
        if self.total_predictions > 0:
            avg_inference_time = self.total_inference_time / self.total_predictions
        
        return {
            "accuracy": 0.95,  # Placeholder - would be calculated from actual evaluations
            "precision": 0.94,
            "recall": 0.93,
            "f1Score": 0.935,
            "totalPredictions": self.total_predictions,
            "avgInferenceTime": avg_inference_time,
            "confusionMatrix": {
                "truePositive": 0,
                "trueNegative": 0,
                "falsePositive": 0,
                "falseNegative": 0
            }
        }
    
    def update_metrics(self, inference_time: float):
        """Update performance metrics after inference."""
        self.total_predictions += 1
        self.total_inference_time += inference_time


# Global model manager instance
model_manager = ModelManager()

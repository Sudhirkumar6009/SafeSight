"""
ONNX Model Loader for Violence Detection
==========================================
High-performance ONNX Runtime inference for real-time violence detection.

Supports both CPU and GPU (CUDA) execution providers for optimal performance.
ONNX provides faster inference than TensorFlow/Keras with lower memory footprint.

Model Architecture (MobileNetV2 + LSTM):
- Input: (batch, 16, 224, 224, 3) - 16 frames of 224x224 RGB
- Output: (batch, 1) - violence probability (sigmoid)
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import cv2
from loguru import logger


class ONNXModelLoader:
    """
    ONNX Runtime model loader for violence detection.
    
    Features:
    - Automatic GPU/CPU provider selection
    - Thread-safe inference
    - Model warmup for optimal performance
    - Graceful fallback if model not found
    """
    
    # Model expects 16 frames of 224x224 RGB images
    EXPECTED_FRAMES = 16
    TARGET_SIZE = (224, 224)
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ONNX model loader.
        
        Args:
            model_path: Path to the .onnx model file. If None, uses default path.
        """
        self.model_path = model_path
        self.session = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.is_loaded = False
        self.execution_provider = "CPU"
        self._lock = threading.Lock()
        self._ort = None
        
        # Load model
        self._load_model()
    
    def _load_onnx_runtime(self):
        """Lazy load ONNX Runtime."""
        if self._ort is None:
            try:
                import onnxruntime as ort
                self._ort = ort
                logger.info(f"ONNX Runtime version: {ort.__version__}")
            except ImportError:
                logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
                raise
        return self._ort
    
    def _load_model(self):
        """Load the ONNX model with optimal execution provider."""
        try:
            if not self.model_path:
                logger.warning("No model path provided")
                return
            
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.warning(f"ONNX model not found at {model_path}")
                return
            
            ort = self._load_onnx_runtime()
            
            # Determine available execution providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            # Select best provider (prefer GPU)
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
                self.execution_provider = "CUDA"
            
            if 'DmlExecutionProvider' in available_providers:
                # DirectML for Windows GPU (AMD/Intel/NVIDIA)
                providers.append('DmlExecutionProvider')
                if self.execution_provider == "CPU":
                    self.execution_provider = "DirectML"
            
            # Always add CPU as fallback
            providers.append('CPUExecutionProvider')
            
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = os.cpu_count() or 4
            sess_options.inter_op_num_threads = 2
            
            # Enable memory optimizations
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            # Create inference session
            logger.info(f"Loading ONNX model from {model_path}...")
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Get input shape and update expected frames if different
            input_shape = self.session.get_inputs()[0].shape
            if input_shape and len(input_shape) == 5 and input_shape[1]:
                if isinstance(input_shape[1], int):
                    self.EXPECTED_FRAMES = input_shape[1]
            
            # Log model info
            actual_provider = self.session.get_providers()[0]
            logger.info(f"✅ ONNX model loaded successfully")
            logger.info(f"  - Execution provider: {actual_provider}")
            logger.info(f"  - Input: {self.input_name} {input_shape}")
            logger.info(f"  - Output: {self.output_name} {self.session.get_outputs()[0].shape}")
            logger.info(f"  - Expected frames: {self.EXPECTED_FRAMES}")
            
            self.is_loaded = True
            
            # Warmup model
            self._warmup()
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
    def _warmup(self):
        """Warmup the model with dummy inference to optimize memory allocation."""
        if not self.is_loaded or self.session is None:
            return
        
        try:
            logger.info("Warming up ONNX model...")
            dummy_input = np.zeros(
                (1, self.EXPECTED_FRAMES, 224, 224, 3),
                dtype=np.float32
            )
            
            # Run a few warmup inferences
            for _ in range(3):
                self.session.run(
                    [self.output_name],
                    {self.input_name: dummy_input}
                )
            
            logger.info("✅ ONNX model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess frames for ONNX model input.
        
        Args:
            frames: List of BGR frames (OpenCV format)
            
        Returns:
            Preprocessed array of shape (1, EXPECTED_FRAMES, 224, 224, 3)
        """
        # Ensure we have exactly EXPECTED_FRAMES
        if len(frames) < self.EXPECTED_FRAMES:
            # Pad with repeated last frame
            frames = list(frames) + [frames[-1]] * (self.EXPECTED_FRAMES - len(frames))
        elif len(frames) > self.EXPECTED_FRAMES:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.EXPECTED_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]
        
        processed = []
        for frame in frames:
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Resize to target size
            resized = cv2.resize(rgb_frame, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            processed.append(normalized)
        
        # Stack: (EXPECTED_FRAMES, 224, 224, 3)
        stacked = np.stack(processed, axis=0)
        
        # Add batch dimension: (1, EXPECTED_FRAMES, 224, 224, 3)
        return np.expand_dims(stacked, axis=0)
    
    def predict(self, frames: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Run inference on frames.
        
        Args:
            frames: List of BGR frames (OpenCV format)
            
        Returns:
            Dictionary with prediction results:
            - violence_score: float (0-1)
            - non_violence_score: float (0-1)
            - is_violent: bool
            - inference_time_ms: float
            - timestamp: str (ISO format)
        """
        if not self.is_loaded or self.session is None:
            return None
        
        with self._lock:
            try:
                # Preprocess frames
                input_data = self.preprocess_frames(frames)
                
                # Run inference
                start_time = time.time()
                outputs = self.session.run(
                    [self.output_name],
                    {self.input_name: input_data}
                )
                inference_time = (time.time() - start_time) * 1000
                
                # Parse output
                prediction = outputs[0]
                
                if prediction.shape[-1] == 2:
                    # Two outputs: [violence, non_violence]
                    violence_score = float(prediction[0][0])
                    non_violence_score = float(prediction[0][1])
                else:
                    # Single output (sigmoid)
                    violence_score = float(prediction[0][0])
                    non_violence_score = 1.0 - violence_score
                
                from datetime import datetime
                
                return {
                    "violence_score": violence_score,
                    "non_violence_score": non_violence_score,
                    "is_violent": violence_score >= 0.50,
                    "inference_time_ms": inference_time,
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": self.execution_provider
                }
                
            except Exception as e:
                logger.error(f"ONNX inference error: {e}")
                return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status information."""
        return {
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path) if self.model_path else None,
            "execution_provider": self.execution_provider,
            "expected_frames": self.EXPECTED_FRAMES,
            "target_size": self.TARGET_SIZE,
            "input_name": self.input_name,
            "output_name": self.output_name
        }


# Singleton instance
_onnx_loader: Optional[ONNXModelLoader] = None


def get_onnx_loader(model_path: Optional[str] = None) -> ONNXModelLoader:
    """
    Get or create the ONNX model loader singleton.
    
    Args:
        model_path: Path to .onnx model file (only used on first call)
        
    Returns:
        ONNXModelLoader instance
    """
    global _onnx_loader
    if _onnx_loader is None:
        _onnx_loader = ONNXModelLoader(model_path)
    return _onnx_loader

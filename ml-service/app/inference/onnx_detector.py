"""
SafeSight ML Service - ONNX Violence Detector
=============================================
Clean, simple violence detection using ONNX Runtime.

Based on the trusted reference implementation:
- Rolling frame queue (16 frames)
- Frame skipping for real-time processing
- Simple preprocessing: resize, BGR->RGB, normalize to [0,1]
- ONNX inference with single sigmoid output
"""

import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from typing import Optional, Dict, Any, List
import os
import time
import logging

logger = logging.getLogger(__name__)


class ONNXViolenceDetector:
    """
    Violence detection using ONNX Runtime.
    
    Clean implementation based on the trusted reference code.
    Input: (1, 16, 224, 224, 3) - batch of 16 RGB frames
    Output: single violence probability (0-1)
    """
    
    # Model configuration - from trusted reference
    SEQ_LEN = 16
    IMG_SIZE = 224
    FRAME_SKIP = 3  # Grab every 3rd frame so 16 frames = ~1.5 seconds of real time
    
    # Class labels - from trusted reference
    CLASS_NORMAL = "Normal"
    CLASS_VIOLENCE = "Violence"
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ONNX violence detector.
        
        Args:
            model_path: Path to the .onnx model file
        """
        self.model_path: Optional[str] = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.is_loaded: bool = False
        self.providers_used: List[str] = []
        
        # Rolling frame queue for live streaming - from trusted reference
        self.frame_queue: deque = deque(maxlen=self.SEQ_LEN)
        self.frame_counter: int = 0
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load the ONNX model.
        
        Args:
            model_path: Path to the .onnx model file
            
        Returns:
            Dict with success status and model info
        """
        try:
            if not os.path.exists(model_path):
                return {
                    "success": False,
                    "error": f"Model file not found: {model_path}"
                }
            
            _, ext = os.path.splitext(model_path)
            if ext.lower() != '.onnx':
                return {
                    "success": False,
                    "error": f"Invalid file type: {ext}. Expected .onnx"
                }
            
            logger.info(f"Loading ONNX model from: {model_path}")
            
            # Create inference session with CPUExecutionProvider (from trusted reference)
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input name - from trusted reference
            self.input_name = self.session.get_inputs()[0].name
            
            # Get actual providers being used
            self.providers_used = list(self.session.get_providers())
            
            # Get model info
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]
            
            self.model_path = model_path
            self.is_loaded = True
            
            logger.info(f"ONNX model loaded successfully!")
            logger.info(f"  Input: {self.input_name} {input_info.shape}")
            logger.info(f"  Output: {output_info.name} {output_info.shape}")
            logger.info(f"  Providers: {self.providers_used}")
            
            return {
                "success": True,
                "message": "Model loaded successfully",
                "model_info": {
                    "path": model_path,
                    "input_name": self.input_name,
                    "input_shape": list(input_info.shape) if input_info.shape else None,
                    "output_name": output_info.name,
                    "output_shape": list(output_info.shape) if output_info.shape else None,
                    "providers": self.providers_used
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.is_loaded = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for the model.
        From trusted reference implementation.
        
        Args:
            frame: BGR frame from OpenCV (H, W, C)
            
        Returns:
            Preprocessed RGB frame (224, 224, 3) as float32 [0, 1]
        """
        # Resize to model input size
        ai_frame = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))
        
        # Convert BGR (OpenCV) to RGB
        ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        ai_frame = np.array(ai_frame, dtype=np.float32) / 255.0
        
        return ai_frame
    
    def predict(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Run violence detection on a batch of frames.
        
        Args:
            frames: List of BGR frames from OpenCV
            
        Returns:
            Prediction result with scores and classification
        """
        if not self.is_loaded or not self.session:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Preprocess frames
        processed = []
        for frame in frames:
            processed.append(self.preprocess_frame(frame))
        
        # Ensure we have exactly SEQ_LEN frames
        if len(processed) < self.SEQ_LEN:
            # Pad with repeated last frame
            while len(processed) < self.SEQ_LEN:
                processed.append(processed[-1])
        elif len(processed) > self.SEQ_LEN:
            # Sample frames uniformly
            indices = np.linspace(0, len(processed) - 1, self.SEQ_LEN, dtype=int)
            processed = [processed[i] for i in indices]
        
        # Create the exact batch shape the ONNX model expects: (1, 16, 224, 224, 3)
        # From trusted reference
        input_tensor = np.expand_dims(np.array(processed), axis=0)
        
        # Fire the ONNX Engine - from trusted reference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse output - single sigmoid value - from trusted reference
        prediction_score = float(outputs[0][0][0])
        
        # Determine classification - from trusted reference
        if prediction_score > 0.50:
            classification = self.CLASS_VIOLENCE
            result_confidence = prediction_score
        else:
            classification = self.CLASS_NORMAL
            result_confidence = 1.0 - prediction_score
        
        return {
            "success": True,
            "classification": classification,
            "confidence": result_confidence,
            "violence_score": prediction_score,
            "normal_score": 1.0 - prediction_score,
            "inference_time_ms": inference_time,
            "frame_count": len(frames)
        }
    
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Run violence detection on a video file.
        Based on trusted reference implementation.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Prediction result with video metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"Failed to open video: {video_path}"
            }
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Collect frames with skipping - from trusted reference
        frames = []
        frame_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            if frame_counter % self.FRAME_SKIP == 0:
                frames.append(frame)
                
                # Stop once we have enough frames
                if len(frames) >= self.SEQ_LEN:
                    break
        
        cap.release()
        
        if len(frames) == 0:
            return {
                "success": False,
                "error": "No frames extracted from video"
            }
        
        # Run prediction
        result = self.predict(frames)
        
        # Add video metadata
        result["video_metadata"] = {
            "path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration
        }
        
        return result
    
    def add_frame_to_queue(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the rolling queue (for live streaming).
        Based on trusted reference implementation.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            True if queue is full and ready for prediction
        """
        self.frame_counter += 1
        
        # Grab every Nth frame - from trusted reference
        if self.frame_counter % self.FRAME_SKIP == 0:
            processed = self.preprocess_frame(frame)
            self.frame_queue.append(processed)
        
        return len(self.frame_queue) == self.SEQ_LEN
    
    def predict_from_queue(self) -> Optional[Dict[str, Any]]:
        """
        Run prediction on the current frame queue.
        Based on trusted reference implementation.
        
        Returns:
            Prediction result or None if queue not full
        """
        if len(self.frame_queue) < self.SEQ_LEN:
            return None
        
        if not self.is_loaded or not self.session:
            return None
        
        start_time = time.time()
        
        # Create the exact batch shape the ONNX model expects: (1, 16, 224, 224, 3)
        # From trusted reference
        input_tensor = np.expand_dims(np.array(self.frame_queue), axis=0)
        
        # Fire the ONNX Engine - from trusted reference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse output - from trusted reference
        prediction_score = float(outputs[0][0][0])
        
        if prediction_score > 0.50:
            classification = self.CLASS_VIOLENCE
            result_confidence = prediction_score
        else:
            classification = self.CLASS_NORMAL
            result_confidence = 1.0 - prediction_score
        
        return {
            "classification": classification,
            "confidence": result_confidence,
            "violence_score": prediction_score,
            "normal_score": 1.0 - prediction_score,
            "inference_time_ms": inference_time
        }
    
    def reset_queue(self):
        """Reset the frame queue."""
        self.frame_queue.clear()
        self.frame_counter = 0
    
    def unload_model(self):
        """Unload the model and free memory."""
        self.session = None
        self.input_name = None
        self.is_loaded = False
        self.providers_used = []
        self.reset_queue()
        logger.info("Model unloaded")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        return {
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "providers": self.providers_used,
            "queue_size": len(self.frame_queue),
            "frame_counter": self.frame_counter,
            "config": {
                "seq_len": self.SEQ_LEN,
                "img_size": self.IMG_SIZE,
                "frame_skip": self.FRAME_SKIP
            }
        }


# Global detector instance
_detector: Optional[ONNXViolenceDetector] = None


def get_detector() -> ONNXViolenceDetector:
    """Get or create the global detector instance."""
    global _detector
    if _detector is None:
        _detector = ONNXViolenceDetector()
    return _detector


def load_detector(model_path: str) -> Dict[str, Any]:
    """Load the detector with the specified model."""
    detector = get_detector()
    return detector.load_model(model_path)

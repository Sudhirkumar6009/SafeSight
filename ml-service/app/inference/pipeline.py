"""
SafeSight ML Service - Inference Pipeline

This module handles the complete inference pipeline for violence detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, Optional
import logging

from ..models import model_manager
from ..utils import load_video_frames, preprocess_frames, analyze_frame_scores
from ..config import settings

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Handles the complete inference pipeline for violence detection.
    """
    
    def __init__(self):
        self.model_manager = model_manager
    
    def predict(
        self,
        video_path: str,
        model_path: Optional[str] = None,
        architecture: Optional[str] = None,
        num_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to the video file
            model_path: Optional path to model (uses loaded model if not specified)
            architecture: Model architecture (if loading new model)
            num_frames: Number of frames to extract
        
        Returns:
            Prediction results dictionary
        """
        start_time = time.time()
        
        try:
            # Load model if path specified and different from current
            if model_path and model_path != self.model_manager.model_path:
                result = self.model_manager.load_model(
                    model_path,
                    architecture or settings.model_architecture
                )
                if not result["success"]:
                    return {
                        "success": False,
                        "error": result.get("error", "Failed to load model")
                    }
            
            # Check if model is loaded
            if not self.model_manager.is_loaded:
                return {
                    "success": False,
                    "error": "No model loaded. Please load a model first."
                }
            
            # Set parameters
            n_frames = num_frames or settings.num_frames
            frame_size = (settings.frame_size, settings.frame_size)
            
            # Load and preprocess video
            logger.info(f"Processing video: {video_path}")
            frames, metadata = load_video_frames(
                video_path,
                num_frames=n_frames,
                frame_size=frame_size
            )
            
            # Preprocess frames
            input_tensor = preprocess_frames(frames)
            
            # Run inference using model manager (handles both PyTorch and Keras)
            if self.model_manager.model_type == "keras":
                # MobileNetV2-LSTM model expects float32 RGB frames
                # The MobileNetLSTMModel wrapper normalizes [0-255] -> [0-1] internally
                # Input shape: (batch, num_frames, 224, 224, 3)
                input_data = frames.astype(np.float32)  # (T, H, W, C) as float32
                input_data = np.expand_dims(input_data, axis=0)  # (1, T, H, W, C)
                
                logger.info(f"Keras input shape: {input_data.shape}, range: [{input_data.min():.1f}, {input_data.max():.1f}]")
                
                # Run prediction - VGG16LSTMModel handles feature extraction
                try:
                    logits = self.model_manager.model.predict(input_data, verbose=0)
                    logger.info(f"Keras output shape: {logits.shape}, values: {logits}")
                except Exception as e:
                    logger.error(f"Keras prediction failed: {e}")
                    raise e
                
                probs = logits[0]  # Get first batch item
                
                # Handle different output formats
                if len(probs.shape) > 1:
                    probs = probs.flatten()
                
                # If model outputs logits (not probabilities), apply softmax
                if len(probs) >= 2 and (probs.sum() > 1.1 or probs.min() < 0):
                    from scipy.special import softmax
                    probs = softmax(probs)
                
                # Handle single output (sigmoid) vs two outputs (softmax)
                if len(probs) == 1:
                    # Single output - sigmoid style
                    violence_prob = float(probs[0])
                    non_violence_prob = 1.0 - violence_prob
                else:
                    # Two outputs - class probabilities
                    violence_prob = float(probs[0])
                    non_violence_prob = float(probs[1])
            else:
                # PyTorch model
                input_tensor = input_tensor.to(self.model_manager.device)
                
                with torch.no_grad():
                    if self.model_manager.use_fp16:
                        with torch.cuda.amp.autocast():
                            logits = self.model_manager.model(input_tensor)
                    else:
                        logits = self.model_manager.model(input_tensor)
                
                # Get probabilities
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                
                # Map to class labels (index 0 = violence, index 1 = non-violence)
                violence_prob = float(probs[0])
                non_violence_prob = float(probs[1])
            
            # Determine classification
            if violence_prob > non_violence_prob:
                classification = "violence"
                confidence = violence_prob
            else:
                classification = "non-violence"
                confidence = non_violence_prob
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Update metrics
            self.model_manager.update_metrics(inference_time)
            
            # Generate frame analysis (simulated for now)
            frame_scores = self._generate_frame_scores(frames, violence_prob)
            frame_analysis = analyze_frame_scores(frame_scores)
            
            logger.info(f"Inference completed: {classification} ({confidence:.2%})")
            
            return {
                "success": True,
                "classification": classification,
                "confidence": confidence,
                "probabilities": {
                    "violence": violence_prob,
                    "nonViolence": non_violence_prob
                },
                "metrics": {
                    "inferenceTime": inference_time,
                    "framesProcessed": n_frames
                },
                "frameAnalysis": frame_analysis,
                "videoMetadata": metadata
            }
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "classification": "non-violence",
                "confidence": 0,
                "probabilities": {
                    "violence": 0,
                    "nonViolence": 0
                },
                "metrics": {
                    "inferenceTime": time.time() - start_time,
                    "framesProcessed": 0
                }
            }
    
    def _generate_frame_scores(
        self,
        frames: np.ndarray,
        overall_prob: float
    ) -> list:
        """
        Generate per-frame scores based on overall probability.
        In a real implementation, this would analyze each frame individually.
        """
        num_frames = len(frames)
        
        # Simulate frame-level scores with some variation around the overall probability
        base_scores = np.random.normal(overall_prob, 0.1, num_frames)
        base_scores = np.clip(base_scores, 0, 1)
        
        return base_scores.tolist()
    
    def batch_predict(
        self,
        video_paths: list,
        **kwargs
    ) -> list:
        """
        Run inference on multiple videos.
        
        Args:
            video_paths: List of video file paths
            **kwargs: Additional arguments passed to predict()
        
        Returns:
            List of prediction results
        """
        results = []
        for path in video_paths:
            result = self.predict(path, **kwargs)
            results.append({
                "video_path": path,
                **result
            })
        return results


# Global inference pipeline instance
inference_pipeline = InferencePipeline()

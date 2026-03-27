"""
SafeSight ML Service - Inference Pipeline
==========================================
Simplified violence detection pipeline using ONNX Runtime.

Based on trusted reference implementation:
- ONNX-only inference (no PyTorch/Keras)
- Simple preprocessing: resize, BGR->RGB, normalize to [0,1]
- 16-frame input, single sigmoid output
"""

import time
import logging
from typing import Dict, Any, List
import os

from .onnx_detector import ONNXViolenceDetector, get_detector
from ..config import settings

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Simplified violence detection pipeline.
    
    Uses ONNX Runtime for fast, portable inference.
    Based on trusted reference implementation.
    """
    
    def __init__(self):
        self.detector = get_detector()
    
    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run violence detection on a video file.
        
        Args:
            video_path: Path to the video file
            **kwargs: Ignored for API compatibility
        
        Returns:
            Prediction results dictionary
        """
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not self.detector.is_loaded:
                model_to_load = settings.default_model_path
                logger.info(f"Loading model from: {model_to_load}")
                result = self.detector.load_model(model_to_load)
                if not result["success"]:
                    return {
                        "success": False,
                        "error": result.get("error", "Failed to load model")
                    }
            
            # Validate video path
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"Video file not found: {video_path}"
                }
            
            # Run prediction
            result = self.detector.predict_video(video_path)
            
            if not result["success"]:
                return result
            
            # Format response (compatible with existing API)
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "classification": result["classification"].lower().replace(" ", "-"),
                "confidence": result["confidence"],
                "probabilities": {
                    "violence": result["violence_score"],
                    "nonViolence": result["normal_score"]
                },
                "metrics": {
                    "inferenceTime": total_time,
                    "inferenceTimeMs": result["inference_time_ms"],
                    "framesProcessed": result["frame_count"]
                },
                "videoMetadata": result.get("video_metadata", {}),
                "frameAnalysis": {
                    "totalFrames": result["frame_count"],
                    "violentFrames": result["frame_count"] if result["violence_score"] > 0.5 else 0,
                    "nonViolentFrames": result["frame_count"] if result["violence_score"] <= 0.5 else 0,
                    "frameScores": [result["violence_score"]] * result["frame_count"]
                }
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
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
    
    def predict_frames(self, frames: List) -> Dict[str, Any]:
        """
        Run violence detection on a list of frames.
        
        Args:
            frames: List of BGR frames from OpenCV
        
        Returns:
            Prediction results
        """
        if not self.detector.is_loaded:
            model_to_load = settings.default_model_path
            result = self.detector.load_model(model_to_load)
            if not result["success"]:
                return result
        
        return self.detector.predict(frames)
    
    def batch_predict(self, video_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Run inference on multiple videos.
        
        Args:
            video_paths: List of video file paths
            **kwargs: Ignored for API compatibility
        
        Returns:
            List of prediction results
        """
        results = []
        for path in video_paths:
            result = self.predict(path)
            results.append({
                "video_path": path,
                **result
            })
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "is_loaded": self.detector.is_loaded,
            "model_path": self.detector.model_path,
            "providers": self.detector.providers_used,
            "config": {
                "seq_len": self.detector.SEQ_LEN,
                "img_size": self.detector.IMG_SIZE,
                "frame_skip": self.detector.FRAME_SKIP
            }
        }


# Global inference pipeline instance
inference_pipeline = InferencePipeline()

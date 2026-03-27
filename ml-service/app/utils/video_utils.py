"""
SafeSight ML Service - Video Processing Utilities

Simple video preprocessing utilities for violence detection inference.
Based on trusted reference implementation - no PyTorch/ImageNet dependencies.
"""

import cv2
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Constants from trusted reference
IMG_SIZE = 224
SEQ_LEN = 16
FRAME_SKIP = 3


def load_video_frames(
    video_path: str,
    num_frames: int = SEQ_LEN,
    frame_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
    frame_skip: int = FRAME_SKIP
) -> Tuple[np.ndarray, dict]:
    """
    Load and preprocess video frames for model inference.
    Based on trusted reference implementation.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: 16)
        frame_size: Target frame size (height, width) (default: 224x224)
        frame_skip: Grab every Nth frame (default: 3)
    
    Returns:
        Tuple of (frames array, video metadata)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    metadata = {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration
    }
    
    # Extract frames with skipping - from trusted reference
    frames = []
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        if frame_counter % frame_skip == 0:
            # Pre-process for the AI - from trusted reference
            ai_frame = cv2.resize(frame, frame_size)
            ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
            ai_frame = np.array(ai_frame, dtype=np.float32) / 255.0
            
            frames.append(ai_frame)
            
            # Stop once we have enough frames
            if len(frames) >= num_frames:
                break
    
    cap.release()
    
    # Pad if needed
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((*frame_size, 3), dtype=np.float32))
    
    frames = np.array(frames[:num_frames])  # (T, H, W, C)
    
    logger.info(f"Extracted {len(frames)} frames from video with {total_frames} total frames")
    
    return frames, metadata


def preprocess_frame(frame: np.ndarray, frame_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """
    Preprocess a single frame for the model.
    From trusted reference implementation.
    
    Args:
        frame: BGR frame from OpenCV (H, W, C)
        frame_size: Target size (height, width)
        
    Returns:
        Preprocessed RGB frame as float32 [0, 1]
    """
    # Resize to model input size
    ai_frame = cv2.resize(frame, frame_size)
    
    # Convert BGR (OpenCV) to RGB
    ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    ai_frame = np.array(ai_frame, dtype=np.float32) / 255.0
    
    return ai_frame


def analyze_frame_scores(
    frame_probs: List[float],
    threshold: float = 0.5
) -> dict:
    """
    Analyze per-frame violence probabilities.
    
    Args:
        frame_probs: List of violence probabilities for each frame
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        Analysis dictionary
    """
    violent_frames = sum(1 for p in frame_probs if p > threshold)
    non_violent_frames = len(frame_probs) - violent_frames
    
    return {
        "totalFrames": len(frame_probs),
        "violentFrames": violent_frames,
        "nonViolentFrames": non_violent_frames,
        "frameScores": frame_probs
    }

"""
SafeSight ML Service - Video Processing Utilities

This module handles video preprocessing for violence detection inference.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    sampling_strategy: str = "uniform"
) -> Tuple[np.ndarray, dict]:
    """
    Load and preprocess video frames for model inference.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        frame_size: Target frame size (height, width)
        sampling_strategy: Frame sampling strategy ('uniform', 'random', 'first')
    
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
    
    # Calculate frame indices to sample
    if sampling_strategy == "uniform":
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
    elif sampling_strategy == "random":
        if total_frames >= num_frames:
            indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
        else:
            indices = np.arange(total_frames)
    else:  # first
        indices = np.arange(min(num_frames, total_frames))
    
    # Extract frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
    
    cap.release()
    
    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*frame_size, 3), dtype=np.uint8))
    
    frames = np.array(frames[:num_frames])  # (T, H, W, C)
    
    logger.info(f"Extracted {len(frames)} frames from video with {total_frames} total frames")
    
    return frames, metadata


def preprocess_frames(
    frames: np.ndarray,
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    Preprocess frames for model inference.
    
    Args:
        frames: Numpy array of frames (T, H, W, C) in RGB, uint8 [0, 255]
        normalize: Whether to normalize using ImageNet stats
        mean: Normalization mean (used only for PyTorch models)
        std: Normalization std (used only for PyTorch models)
    
    Returns:
        Preprocessed tensor of shape (1, C, T, H, W)
    """
    # Convert to float32 and scale to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    
    # Normalize with explicit float32 arrays
    if normalize:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        frames = (frames - mean) / std
    
    # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
    frames = np.transpose(frames, (3, 0, 1, 2))
    
    # Add batch dimension: (C, T, H, W) -> (1, C, T, H, W)
    tensor = torch.from_numpy(frames).float()  # Ensure float32
    tensor = tensor.unsqueeze(0)
    
    return tensor


def extract_frame_features(
    frames: np.ndarray,
    model: torch.nn.Module,
    device: torch.device
) -> np.ndarray:
    """
    Extract per-frame features using the model backbone.
    
    Args:
        frames: Preprocessed frames tensor
        model: The classification model
        device: Computation device
    
    Returns:
        Feature array for each frame
    """
    # This is a placeholder - actual implementation depends on model architecture
    return np.zeros((len(frames), 512))


def analyze_frame_scores(
    frame_probs: List[float],
    threshold: float = 0.5
) -> dict:
    """
    Analyze per-frame violence probabilities.
    
    Args:
        frame_probs: List of violence probabilities for each frame
        threshold: Classification threshold
    
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

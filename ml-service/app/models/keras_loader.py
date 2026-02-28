"""
Keras Model Loader with compatibility handling for H5 models.

The violence_model_legacy.h5 contains a full end-to-end model:
  - TimeDistributed(MobileNetV2) as feature extractor
  - TimeDistributed(GlobalAveragePooling2D)
  - LSTM(64)
  - Dense(64, relu)
  - Dense(1, sigmoid)

Input shape: (batch, 16, 224, 224, 3) - 16 frames of 224x224 RGB
Output: (batch, 1) - violence probability (sigmoid)
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_keras_model_compatible(model_path: str):
    """
    Load the violence detection Keras model.
    
    The model is a complete Keras Functional model saved with all weights
    (MobileNetV2 + LSTM classifier). We load it directly using 
    tf.keras.models.load_model with compile=False.
    
    Args:
        model_path: Path to .h5 model file
        
    Returns:
        A wrapper model with a .predict() method and .input_shape attribute.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    import tensorflow as tf
    from tensorflow import keras
    
    logger.info(f"Loading Keras model from: {model_path}")
    
    # The H5 file contains the full model architecture + weights
    # Load directly - no need to rebuild the architecture manually
    try:
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"Model loaded successfully via load_model")
        logger.info(f"Input shape: {model.input_shape}")
        logger.info(f"Output shape: {model.output_shape}")
    except Exception as e:
        logger.warning(f"load_model failed ({e}), trying weight-based loading...")
        model = _load_by_rebuilding(model_path)
    
    # Wrap in a simple container for API compatibility
    class MobileNetLSTMModel:
        """Wrapper for the MobileNetV2+LSTM violence detection model."""
        
        def __init__(self, keras_model):
            self._model = keras_model
            # Model expects: (batch, 16, 224, 224, 3)
            self.input_shape = keras_model.input_shape
            raw_frames = self.input_shape[1] if self.input_shape[1] else 16
            self.expected_frames = raw_frames
        
        def predict(self, frames_batch, verbose=0):
            """
            Run inference on a batch of frame sequences.
            
            Args:
                frames_batch: (batch, num_frames, 224, 224, 3) - float32 [0-255] or [0-1]
                verbose: Verbosity level
            
            Returns:
                Violence predictions (batch, 1) - sigmoid output
            """
            import numpy as np
            
            batch_size = frames_batch.shape[0]
            num_frames = frames_batch.shape[1]
            
            # Handle frame count mismatch
            if num_frames < self.expected_frames:
                # Pad by repeating last frame
                padding = np.repeat(
                    frames_batch[:, -1:, :, :, :],
                    self.expected_frames - num_frames,
                    axis=1
                )
                frames_batch = np.concatenate([frames_batch, padding], axis=1)
                logger.debug(f"Padded frames from {num_frames} to {self.expected_frames}")
            elif num_frames > self.expected_frames:
                # Sample frames uniformly
                indices = np.linspace(0, num_frames - 1, self.expected_frames, dtype=int)
                frames_batch = frames_batch[:, indices, :, :, :]
                logger.debug(f"Sampled frames to {self.expected_frames}")
            
            frames_batch = frames_batch.astype(np.float32)
            
            # Normalize to [0, 1] if input is in [0, 255] range.
            # The model was trained with rescale=1./255 normalization.
            # Detect range by checking if max value exceeds 1.0.
            if frames_batch.max() > 1.0:
                frames_batch = frames_batch / 255.0
            
            # Run prediction
            return self._model.predict(frames_batch, verbose=verbose)
    
    return MobileNetLSTMModel(model)


def _load_by_rebuilding(model_path: str):
    """
    Fallback: Rebuild the model architecture and load weights.
    
    Architecture from H5 inspection:
      Input(16, 224, 224, 3) 
      -> TimeDistributed(MobileNetV2) 
      -> TimeDistributed(GlobalAveragePooling2D)
      -> LSTM(64) 
      -> Dense(64, relu) 
      -> Dense(1, sigmoid)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    
    EXPECTED_FRAMES = 16
    
    logger.info("Rebuilding model architecture for weight loading...")
    
    # Build MobileNetV2 feature extractor
    mobilenet = MobileNetV2(
        weights=None,  # We'll load from H5
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Build the full model
    inp = keras.Input(shape=(EXPECTED_FRAMES, 224, 224, 3))
    x = layers.TimeDistributed(mobilenet)(inp)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inp, outputs=out)
    
    # Load weights
    logger.info(f"Loading weights from {model_path}...")
    model.load_weights(model_path)
    logger.info("Weights loaded successfully via rebuild method")
    
    return model

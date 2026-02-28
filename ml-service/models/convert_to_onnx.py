"""Convert Keras H5 model to ONNX format."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
import h5py
import tf_keras as keras
import tensorflow as tf
import tf2onnx
import onnx

def fix_config(config):
    """Fix legacy Keras config issues."""
    if isinstance(config, dict):
        # Fix batch_shape -> batch_input_shape for InputLayer
        if config.get('class_name') == 'InputLayer':
            if 'batch_shape' in config.get('config', {}):
                config['config']['batch_input_shape'] = config['config'].pop('batch_shape')
        
        # Fix nested dtype policy (convert dict to just the dtype string)
        if 'config' in config and isinstance(config['config'], dict):
            inner_config = config['config']
            if 'dtype' in inner_config and isinstance(inner_config['dtype'], dict):
                dtype_config = inner_config['dtype']
                if 'config' in dtype_config and 'name' in dtype_config['config']:
                    inner_config['dtype'] = dtype_config['config']['name']
        
        # Recursively fix nested configs
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                fix_config(value)
    elif isinstance(config, list):
        for item in config:
            fix_config(item)
    return config

def convert_keras_to_onnx(h5_path: str, onnx_path: str):
    """Convert a Keras .h5 model to ONNX format."""
    print(f"Loading model from {h5_path}...")
    
    # Load H5 file and fix config
    with h5py.File(h5_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError("No model config found in H5 file")
        
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        
        config = json.loads(model_config)
        config = fix_config(config)
        
        print("Fixed config, rebuilding model...")
        
    # Rebuild model from fixed config
    model = keras.models.model_from_config(config)
    
    # Load weights
    model.load_weights(h5_path)
    
    print("Model loaded successfully.")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Get input signature
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
    
    print("Converting to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    
    # Save the ONNX model
    import onnx
    onnx.save(onnx_model, onnx_path)
    print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    convert_keras_to_onnx("violence_model_legacy.h5", "model.onnx")

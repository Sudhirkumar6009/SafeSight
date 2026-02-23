"""Test loading the Keras MobileNetV2+LSTM model"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

def write_result(msg):
    """Write to a file so output is captured regardless of redirects"""
    with open('test_load_result.txt', 'a') as f:
        f.write(msg + '\n')

# Clear previous results
if os.path.exists('test_load_result.txt'):
    os.remove('test_load_result.txt')

try:
    import warnings
    warnings.filterwarnings('ignore')
    
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    write_result(f"TensorFlow version: {tf.__version__}")
    
    # Use the proper compatibility loader
    sys.path.insert(0, os.path.dirname(__file__))
    from app.models.keras_loader import load_keras_model_compatible
    
    MODEL_PATH = './models/violence_model_legacy.h5'
    
    if not os.path.exists(MODEL_PATH):
        write_result(f"ERROR: Model file not found at {MODEL_PATH}")
        write_result(f"Current directory: {os.getcwd()}")
        if os.path.exists('models'):
            write_result(f"Models directory contents: {os.listdir('models')}")
        print("Check test_load_result.txt for results")
        sys.exit(1)
    
    try:
        model = load_keras_model_compatible(MODEL_PATH)
        
        write_result("=" * 50)
        write_result("SUCCESS! MobileNetV2+LSTM Model loaded")
        write_result("=" * 50)
        write_result(f"Model type: {type(model).__name__}")
        write_result(f"Model input shape: {model.input_shape}")
        write_result(f"Expected frames: {model.expected_frames}")
        
        # Test with dummy data (16 frames of 224x224 RGB)
        import numpy as np
        dummy_input = np.random.randint(0, 255, (1, 16, 224, 224, 3)).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        write_result(f"Test prediction shape: {prediction.shape}")
        write_result(f"Test prediction value: {prediction[0][0]:.4f}")
        write_result("Model inference test PASSED!")
        
        print("SUCCESS - Check test_load_result.txt for results")
        
    except Exception as e:
        write_result("=" * 50)
        write_result(f"ERROR: {type(e).__name__}")
        write_result("=" * 50)
        write_result(f"Message: {str(e)}")
        import traceback
        write_result(traceback.format_exc())
        print("FAILED - Check test_load_result.txt for results")
        
except Exception as e:
    write_result(f"Import Error: {str(e)}")
    print("FAILED - Check test_load_result.txt for results")

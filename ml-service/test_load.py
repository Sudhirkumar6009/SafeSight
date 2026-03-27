"""Test loading the ONNX violence detection model"""
import os
import sys

def write_result(msg):
    """Write to a file so output is captured regardless of redirects"""
    with open('test_load_result.txt', 'a') as f:
        f.write(msg + '\n')
    print(msg)

# Clear previous results
if os.path.exists('test_load_result.txt'):
    os.remove('test_load_result.txt')

try:
    import numpy as np
    import onnxruntime as ort
    
    write_result(f"ONNX Runtime version: {ort.__version__}")
    
    MODEL_PATH = './models/violence_model.onnx'
    
    if not os.path.exists(MODEL_PATH):
        write_result(f"ERROR: Model file not found at {MODEL_PATH}")
        write_result(f"Current directory: {os.getcwd()}")
        if os.path.exists('models'):
            write_result(f"Models directory contents: {os.listdir('models')}")
        sys.exit(1)
    
    try:
        # Load ONNX model - from trusted reference
        write_result(f"Loading ONNX model from: {MODEL_PATH}")
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        write_result("=" * 50)
        write_result("SUCCESS! ONNX Model loaded")
        write_result("=" * 50)
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        write_result(f"Input name: {input_info.name}")
        write_result(f"Input shape: {input_info.shape}")
        write_result(f"Output name: {output_info.name}")
        write_result(f"Output shape: {output_info.shape}")
        write_result(f"Providers: {session.get_providers()}")
        
        # Test with dummy data (1, 16, 224, 224, 3) - from trusted reference
        write_result("\nRunning test inference...")
        dummy_input = np.random.rand(1, 16, 224, 224, 3).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        
        prediction_score = outputs[0][0][0]
        write_result(f"Test prediction score: {prediction_score:.4f}")
        
        if prediction_score > 0.5:
            write_result(f"Classification: Violence ({prediction_score*100:.1f}%)")
        else:
            write_result(f"Classification: Normal ({(1-prediction_score)*100:.1f}%)")
        
        write_result("\nModel inference test PASSED!")
        
    except Exception as e:
        write_result("=" * 50)
        write_result(f"ERROR: {type(e).__name__}")
        write_result("=" * 50)
        write_result(f"Message: {str(e)}")
        import traceback
        write_result(traceback.format_exc())
        
except Exception as e:
    write_result(f"Import Error: {str(e)}")
    import traceback
    write_result(traceback.format_exc())

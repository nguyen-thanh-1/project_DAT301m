import sys
import os

print("=== Python and Environment Info ===")
print(f"Python Version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Conda Prefix: {os.environ.get('CONDA_PREFIX', 'Not in a conda env')}")

print("\n=== TensorFlow Status ===")
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) Detected: {len(gpus)}")
        for gpu in gpus:
            print(f" - {gpu}")
        
        # Test a simple operation on GPU
        print("\n=== Running GPU Verification Test ===")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("TensorFlow GPU Matrix Multiplication Test: SUCCESS")
                print(f"Result:\n{c}")
        except Exception as e:
            print(f"TensorFlow GPU Matrix Multiplication Test: FAILED\nError: {e}")
    else:
        print("GPU(s) Detected: None (TensorFlow is running on CPU)")
        
except ImportError:
    print("TensorFlow is NOT installed in this environment.")
    print("Please install it using: uv pip install tensorflow==2.10.0")

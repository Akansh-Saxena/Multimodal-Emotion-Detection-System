import os
import sys

def check_gpu():
    print("=== GPU Configuration Check ===")
    
    # 1. PyTorch Setup
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA Available: YES")
            print(f"PyTorch CUDA Device Count: {torch.cuda.device_count()}")
            print(f"PyTorch Current Device: {torch.cuda.get_device_name(0)}")
        else:
            print("PyTorch CUDA Available: NO (Falling back to CPU)")
    except ImportError:
        print("PyTorch not installed.")
        
    print("-" * 30)
    
    # 2. TensorFlow Setup
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow GPU Available: YES")
            for i, gpu in enumerate(gpus):
                print(f"TensorFlow GPU {i}: {gpu.name}")
            
            # Prevent TF from taking all VRAM
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("TensorFlow memory growth set to True.")
            except RuntimeError as e:
                print(e)
        else:
            print("TensorFlow GPU Available: NO (Falling back to CPU)")
    except ImportError:
        print("TensorFlow not installed.")

if __name__ == "__main__":
    check_gpu()

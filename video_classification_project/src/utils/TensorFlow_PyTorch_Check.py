import tensorflow as tf
import torch

print("=== TensorFlow Check ===")
print("TensorFlow version:", tf.__version__)
gpus_tf = tf.config.list_physical_devices('GPU')
if gpus_tf:
    print("✅ TensorFlow GPU available:", gpus_tf)
else:
    print("❌ No GPU detected for TensorFlow. Using CPU.")

print("\n=== PyTorch Check ===")
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("✅ PyTorch GPU available")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version (PyTorch compiled with):", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())
else:
    print("❌ No GPU detected for PyTorch. Using CPU.")

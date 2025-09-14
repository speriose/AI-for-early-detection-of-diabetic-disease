import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# List physical devices (GPUs)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Check if TensorFlow is using GPU
if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA support.")
else:
    print("TensorFlow is NOT built with CUDA support.")

if tf.test.is_gpu_available():
    print("GPU is available and being used.")
else:
    print("GPU is not available or not being used.")

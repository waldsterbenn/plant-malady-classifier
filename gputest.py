import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available and TensorFlow is using it:")
    print(physical_devices)
else:
    print("GPU is NOT available to TensorFlow. Check installation steps.")

# Optional: Run a simple operation on the GPU to confirm
try:
    with tf.device('/GPU:1'):  # Try to run on the first GPU
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    print("GPU computation successful!")
except tf.errors.NotFoundError:
    print("GPU computation failed.  GPU device not found.")

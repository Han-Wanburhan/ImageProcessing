import tensorflow as tf
if tf.test.is_gpu_available:
    print("GPU")
else:
    print("CPU")    
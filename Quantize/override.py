import tensorflow as tf
from TensorQuant.Quantize.override_functions import generic_keras_override

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# override these qmaps in your main application
# Example:
# intr_q_map = {    "MyNetwork/Conv2D_1" : "nearest,32,16",
#                   "MyNetwork/Conv2D_2" : "nearest,16,8"}

intr_q_map=None
extr_q_map=None
weight_q_map=None
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

tf.python.keras.layers.VersionAwareLayers = tf.keras.layers.Layer

# 'tensorflow.keras.layers.Convolution2D' override
keras_conv2d = tf.keras.layers.Conv2D
keras_conv2d_override = generic_keras_override(keras_conv2d)
#tf.keras.layers.Conv2D = keras_conv2d_override
tf.keras.layers.Conv2D = keras_conv2d_override

# 'tf.keras.layers.Conv1D' override
keras_conv1d = tf.keras.layers.Conv1D
keras_conv1d_override = generic_keras_override(keras_conv1d)
tf.keras.layers.Conv1D = keras_conv1d_override

# 'tf.keras.layers.Dense' override
keras_dense = tf.keras.layers.Dense
keras_dense_override = generic_keras_override(keras_dense)
tf.keras.layers.Dense = keras_dense_override

# 'tf.keras.layers.MaxPooling2D' override
keras_maxpool2d = tf.keras.layers.MaxPooling2D
keras_maxpool2d_override = generic_keras_override(keras_maxpool2d)
tf.keras.layers.MaxPooling2D = keras_maxpool2d_override

# 'tf.keras.layers.MaxPool1D' override
keras_maxpool1d = tf.keras.layers.MaxPool1D
keras_maxpool1d_override = generic_keras_override(keras_maxpool1d)
tf.keras.layers.MaxPool1D = keras_maxpool1d_override

keras_zero2d = tf.keras.layers.ZeroPadding2D
keras_zero2d_override = generic_keras_override(keras_zero2d)
tf.keras.layers.ZeroPadding2D = keras_zero2d_override

keras_BN = tf.keras.layers.BatchNormalization
keras_BN_override = generic_keras_override(keras_BN)
tf.keras.layers.BatchNormalization = keras_BN_override

keras_ReLU = tf.keras.layers.ReLU
keras_ReLU_override = generic_keras_override(keras_ReLU)
tf.keras.layers.ReLU = keras_ReLU_override

keras_globalA2D = tf.keras.layers.GlobalAveragePooling2D
keras_globalA2D_override = generic_keras_override(keras_globalA2D)
tf.keras.layers.GlobalAveragePooling2D = keras_globalA2D_override
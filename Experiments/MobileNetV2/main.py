#import sys
#sys.path.append('..')

# LeNet for MNIST using Keras and TensorFlow
import tensorflow as tf
import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import numpy as np



# Add this for TensorQuant
from TensorQuant.Quantize import override

from model import MobileNetV2

def main():
    perturbed_layer_name = 'block_16_project' #Get layer names: print(mode.layers[i].name)
    override.extr_q_map={"Conv1" : "binary", "bn_Conv1" : "binary", "Conv1_relu": "binary"}
    override.weight_q_map={"Conv1" : "binary", "bn_Conv1" : "binary", "Conv1_relu": "binary"}
    img_shape = (224,224,3)
    batch_size = (32,)
    batch_shape = batch_size+ img_shape

    #First we get two models
    perturbed_model = MobileNetV2(input_shape=img_shape,
                                    include_top=True,
                                    weights='imagenet')
    #Generate some test data
    batch = mobilenet_v2.preprocess_input(np.random.random_sample(batch_shape))

    # Get intermediate outputs of perturbed and unperturbed and compare
    # https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction


    perturbed_intermediate_layer_model = tf.keras.Model(inputs=perturbed_model.input,
                            outputs=perturbed_model.get_layer(perturbed_layer_name).output)
    perturbed_intermediate_output = perturbed_intermediate_layer_model(batch)

    # Get global outouts
    perturbed_output = perturbed_model(batch)
    print(np.linalg.norm(np.array(perturbed_output)))
    

if __name__ == "__main__":
    main()

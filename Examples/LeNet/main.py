#import sys
#sys.path.append('..')

# LeNet for MNIST using Keras and TensorFlow
import tensorflow as tf
import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import numpy as np

import model

# Add this for TensorQuant
from TensorQuant.Quantize import override

def main():
    # Control which devices TF sees. '-1' = None, '0', '1','2,'3'...PCI Bus ID
    # https://www.tensorflow.org/guide/gpu
    # https://github.com/tensorflow/tensorflow/issues/24496
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Controll how much and how TF allocates GPU memory
    # https://www.tensorflow.org/guide/gpu
    # https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
    # Option 1: Allow memory growth. This means at the beginning, only a tiny fraction allocated, but memory consumption grows with process
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Option 2: set a limit on what TF trys to allocate per process
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
    #    try:
    #        for gpu in gpus:
    #            tf.config.experimental.set_virtual_device_configuration(gpu, [
    #                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]) #MB, maximum 14235 per device
    #    except RuntimeError as e:
    #        print(e)

    # TensorQuant
    # Make sure the overrides are set before the model is created!
    # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    override.extr_q_map={"Conv1" : "nearest,12,11"}
    override.weight_q_map={ "Conv1" : "nearest,32,16", "Dense3" : "nearest,32,16"}
    # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

    # Download the MNIST dataset
    dataset = mnist.load_data()

    train_data = dataset[0][0]
    train_labels = dataset[0][1]

    test_data = dataset[1][0]
    test_labels = dataset[1][1]

    # Reshape the data to a (70000, 28, 28, 1) tensord
    train_data = train_data.reshape([*train_data.shape,1]) / 255.0

    test_data = test_data.reshape([*test_data.shape,1]) / 255.0

    # Tranform training labels to one-hot encoding
    train_labels = np.eye(10)[train_labels]

    # Tranform test labels to one-hot encoding
    test_labels = np.eye(10)[test_labels]

    test_accuracies = list()
    test_losses = list()

    for i in range(100):
        lenet = model.LeNet()

        lenet.summary()

        optimizer = tf.keras.optimizers.SGD(lr=0.01)

        # Compile the network
        lenet.compile(
            loss = "categorical_crossentropy",
            optimizer = optimizer,
            metrics = ["accuracy"])

        # Callbacks
        callbacks_list=[]
        #callbacks_list.append(callbacks.WriteTrace("timeline_%02d.json"%(myRank), run_metadata) )

        # Train the model
        lenet.fit(
            train_data,
            train_labels,
            batch_size = 128,
            epochs = 1,
            verbose = 1,
            callbacks=callbacks_list)

        # Evaluate the model
        (loss, accuracy) = lenet.evaluate(
            test_data,
            test_labels,
            batch_size = 128,
            verbose = 1)
        # Push the model's accuracy in list
        test_accuracies.append(accuracy)
        test_losses.append(loss)

    test_accuracies, test_losses = (list(x) for x in zip(*sorted(zip(test_accuracies, test_losses), key=lambda pair: pair[0])))

    trimmed_mean_accuracy = 0
    trimmed_mean_loss = 0

    for i in range(2,98):
        trimmed_mean_accuracy += test_accuracies[i]
        trimmed_mean_loss += test_losses[i]

    trimmed_mean_accuracy /= 95
    trimmed_mean_loss /= 95

    print(test_accuracies)
    print(test_losses)

    print(trimmed_mean_accuracy)
    print(trimmed_mean_loss)

if __name__ == "__main__":
        main()

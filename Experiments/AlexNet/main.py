#import sys
#sys.path.append('..')

# AlexNet for Flowers using Keras and TensorFlow
import tensorflow as tf
import os

#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.datasets import tf_flowers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    #override.extr_q_map={"Conv1" : "nearest,12,11"}
    #override.weight_q_map={ "Conv1" : "nearest,32,16", "Dense3" : "nearest,32,16"}
    # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

    # Preprocess the flowers dataset
    # Source:
    # https://www.kaggle.com/anandsingh1011/flower-recognition-model-in-keras/data
    script_dir = os.path.dirname(".")
    training_set_path = os.path.join(script_dir, './input/flowers/flowers/')
    test_set_path = os.path.join(script_dir, './input/flowers/flowers/')
    batch_size = 128
    input_size = (256, 256)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.33)

    test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.33)

    training_set = train_datagen.flow_from_directory(training_set_path,
                                                     target_size=input_size,
                                                     batch_size=batch_size,
                                                     subset="training",
                                                     class_mode='categorical')

    validation_set = train_datagen.flow_from_directory(test_set_path,
                                                target_size=input_size,
                                                batch_size=batch_size,
                                                subset="validation",
                                                class_mode='categorical')

    '''dataset = tf_flowers.load_data()
    '''
    #train_data = training_set[0]
    #train_labels = training_set[1]

    #test_data = test_set[0]
    #test_labels = test_set[1]
    '''
    # Reshape the data to a (70000, 28, 28, 1) tensord
    train_data = train_data.reshape([*train_data.shape,3]) / 255.0

    test_data = test_data.reshape([*test_data.shape,3]) / 255.0

    # Transform training labels to one-hot encoding
    train_labels = np.eye(5)[train_labels]

    # Transform test labels to one-hot encoding
    test_labels = np.eye(5)[test_labels]
    '''

    alexnet = model.AlexNet()

    alexnet.summary()

    optimizer = tf.keras.optimizers.SGD(lr=0.01)

    # Compile the network
    alexnet.compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer,
        metrics = ["accuracy"])

    # Callbacks
    callbacks_list=[]
    #callbacks_list.append(callbacks.WriteTrace("timeline_%02d.json"%(myRank), run_metadata) )

    # Train the model
    hist = alexnet.fit_generator(
        training_set,
        steps_per_epoch= training_set.samples // batch_size,
        validation_steps = validation_set.samples // batch_size,
        validation_data= validation_set,
        epochs = 1,
        verbose = 1,
        callbacks=callbacks_list)

    # Evaluate the model
    (loss, accuracy) = alexnet.evaluate(
        validation_set,
        steps = validation_set.samples // batch_size,
        verbose = 1)
    # Print the model's accuracy
    #print("Test accuracy: %.2f"%(accuracy))

if __name__ == "__main__":
    main()

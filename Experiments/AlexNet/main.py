#import sys
#sys.path.append('..')

# AlexNet for Flowers using Keras and TensorFlow
import tensorflow as tf
import os

#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.datasets import tf_flowers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from operator import add
import model

import pickle
import timeit

# Add this for TensorQuant
from TensorQuant.Quantize import override

def main():
    # Control which devices TF sees. '-1' = None, '0', '1','2,'3'...PCI Bus ID
    # https://www.tensorflow.org/guide/gpu
    # https://github.com/tensorflow/tensorflow/issues/24496
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'

    # Controll how much and how TF allocates GPU memory
    # https://www.tensorflow.org/guide/gpu
    # https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
    # Option 1: Allow memory growth. This means at the beginning, only a tiny fraction allocated, but memory consumption grows with process
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
#if gpus:
    #    try:
    #        for gpu in gpus:
    #            tf.config.experimental.set_memory_growth(gpu, True)
    #    except RuntimeError as e:
    #        print(e)

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
    #override.weight_q_map={ "Conv1" : "nearest,16,8"}
    # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

    # Preprocess the flowers dataset
    # Source:
    # https://www.kaggle.com/anandsingh1011/flower-recognition-model-in-keras/data
    script_dir = os.path.dirname(".")
    training_set_path = os.path.join(script_dir, './input/flowers/flowers/')
    test_set_path = os.path.join(script_dir, './input/flowers/flowers/')
    batch_size = 128
    num_epochs = 80
    num_runs = 10
    input_size = (256, 256)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.33)

    #test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.33)

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


    test_accuracies = list()
    test_losses = list()
    test_time = list()
    avg_hist_acc = None
    avg_hist_acc_val = None
    for i in range(10):
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

        start = timeit.default_timer()

        # Train the model
        hist = alexnet.fit_generator( #history can later be used to get convergence rates
            training_set,
            steps_per_epoch= training_set.samples // batch_size,
            validation_steps = validation_set.samples // batch_size,
            validation_data= validation_set,
            epochs = num_epochs,
            verbose = 1,
            callbacks=callbacks_list)

        stop = timeit.default_timer()
        test_time.append(stop - start)

        # Evaluate the model
        (loss, accuracy) = alexnet.evaluate(
            validation_set,
            steps = validation_set.samples // batch_size,
            verbose = 1)
        # Print the model's accuracy
        # print("Test accuracy: %.2f"%(accuracy))
        test_accuracies.append(accuracy)
        test_losses.append(loss)

        if avg_hist_acc is None and avg_hist_acc_val is None:
            avg_hist_acc = hist.history['accuracy']
            avg_hist_acc_val = hist.history['val_accuracy']
        else:
            avg_hist_acc = list(map(add, avg_hist_acc, hist.history['accuracy']))
            avg_hist_acc_val = list(map(add, avg_hist_acc, hist.history['val_accuracy']))

    avg_hist_acc = [x * 1/num_runs for x in avg_hist_acc]
    avg_hist_acc_val = [x * 1 / num_runs for x in avg_hist_acc_val]

    test_accuracies, test_losses = (list(x) for x in
                                    zip(*sorted(zip(test_accuracies, test_losses), key=lambda pair: pair[0])))

    trimmed_mean_accuracy = 0
    trimmed_mean_loss = 0
    trimmed_mean_time = 0

    for i in range(1, 9):
        trimmed_mean_accuracy += test_accuracies[i]
        trimmed_mean_loss += test_losses[i]
        trimmed_mean_time += test_time[i]
        trimmed_mean_time = test_time[i]

    trimmed_mean_accuracy /= 8
    trimmed_mean_loss /= 8
    trimmed_mean_time /= 8

    results = {}
    results['eval_accuracies'] = test_accuracies
    results['eval_losses'] = test_losses
    results['eval_trimmed_mean_accuracy'] = trimmed_mean_accuracy
    results['eval_trimmed_mean_loss'] = trimmed_mean_loss
    results['avg_train_trimmed_mean_time'] = trimmed_mean_time
    results['avg_train_hist_acc'] = avg_hist_acc
    results['avg_train_hist_acc_val'] = avg_hist_acc_val

    with open('/storage/results.pkl', 'wb') as fp:
        pickle.dump(results, fp)

    #print('Test history avg, acc, val_acc')
    #print(avg_hist_acc)
    #print(avg_hist_acc_val)

    #print('Test accuracies avg, acc, loss')
    #print(test_accuracies)
    #print(test_losses)

    #print(trimmed_mean_accuracy)
    #print(trimmed_mean_loss)

if __name__ == "__main__":
    main()

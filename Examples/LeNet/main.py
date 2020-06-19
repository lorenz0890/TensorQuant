#import sys
#sys.path.append('..')

# LeNet for MNIST using Keras and TensorFlow
import tensorflow as tf
import os
from operator import add

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import numpy as np

import pickle

import model
import timeit


from tensorflow import keras # Required for Tensorboard
from datetime import datetime # Required for Tensorboard

# Add this for TensorQuant
from TensorQuant.Quantize import override
from TensorQuant.Quantize.Quantizers import Quantizer_Reference

def main():
    # Control which devices TF sees. '-1' = None, '0', '1','2,'3'...PCI Bus ID
    # https://www.tensorflow.org/guide/gpu
    # https://github.com/tensorflow/tensorflow/issues/24496
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Controll how much and how TF allocates GPU memory
    # https://www.tensorflow.org/guide/gpu
    # https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
    # Option 1: Allow memory growth. This means at the beginning, only a tiny fraction allocated, but memory consumption grows with process
    #gpus = tf.config.experimental.list_physical_devices('GPU')
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
    #override.weight_q_map={ "Dense3" : Quantizer_Reference(16,8)}
    #override.weight_q_map={ "Dense3" : "nearest,16,8"}
    #override.weight_q_map = {"Conv1": "binary", "MaxPool1": "binary", "Conv2": "binary", "MaxPool2": "binary", "Dense3": "binary", "Dense4": "binary"}

    # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

    # Global Variable num_epochs, num_runs
    num_epochs = 80
#    num_runs = 1
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

#    test_accuracies = 0
#    test_losses = 0
#    test_time = 0
#    avg_hist_acc = None
#    avg_hist_acc_val = None

#    for i in range(num_runs):
#    tf.summary.trace_on(graph=True, profiler=True)  # Required by Tensorboard Profiling
    lenet = model.LeNet()
#        lenet.summary()
    optimizer = tf.keras.optimizers.SGD(lr=0.01)
        # Compile the network
    lenet.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
        run_eagerly=False
    )
        # Callbacks
    callbacks_list=[]
        #callbacks_list.append(callbacks.WriteTrace("timeline_%02d.json"%(myRank), run_metadata) )
#        logdir = "/storage/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback_scalars = keras.callbacks.TensorBoard(log_dir=logdir)

    logdir2 = "/storage/logs/performance/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback_perf = tf.keras.callbacks.TensorBoard(log_dir = logdir2,
                                                         histogram_freq = 1,
                                                         profile_batch = 0) # Profile Batch can be changed

        # Warning:
        # callbacks might affect external runtime measurements with timeit and such
        # profiling also uses a lot of memorynnen
        #callbacks_list.append(tensorboard_callback_scalars) # comment this to deactivate TB scalars
    callbacks_list.append(tensorboard_callback_perf) # comment this to deactivate TB profiling

        # Start Timer
    start = timeit.default_timer()

        # Train the model
    hist = lenet.fit(
        train_data,
        train_labels,
        batch_size=128,
        epochs=num_epochs,
        validation_split=0.33,
        verbose=1,
        callbacks=callbacks_list)

#        stop = timeit.default_timer()

#        test_time = stop - start

        # Evaluate the model
#        (loss, accuracy) = lenet.evaluate(
#            test_data,
#            test_labels,
#            batch_size=128,
#            verbose=1)
        # Push the model's accuracy in list
#        test_accuracies = accuracy
#        test_losses = loss

#        if avg_hist_acc is None and avg_hist_acc_val is None:
#            avg_hist_acc = hist.history['accuracy']
#            avg_hist_acc_val = hist.history['val_accuracy']
#        else:
#            avg_hist_acc = list(map(add, avg_hist_acc, hist.history['accuracy']))
#            avg_hist_acc_val = list(map(add, avg_hist_acc_val, hist.history['val_accuracy']))

#    avg_hist_acc = [x * (1 / num_runs) for x in avg_hist_acc]
#    avg_hist_acc_val = [x * (1 / num_runs) for x in avg_hist_acc_val]

    #test_accuracies, test_losses, test_time = (list(x) for x in
                                    #zip(*sorted(zip(test_accuracies, test_losses, test_time), key=lambda pair: pair[0])))

    #trimmed_mean_accuracy = 0
    #trimmed_mean_loss = 0
    #trimmed_mean_time = 0

    #for i in range(1, 9):
    #    trimmed_mean_accuracy += test_accuracies[i]
    #    trimmed_mean_loss += test_losses[i]
    #    trimmed_mean_time += test_time[i]

    #trimmed_mean_accuracy /= 8
    #trimmed_mean_loss /= 8
    #trimmed_mean_time /= 8

#    results = {}
#    results['eval_accuracies'] = test_accuracies
#    results['eval_losses'] = test_losses
#    results['eval_time'] = test_time
    #esults['eval_trimmed_mean_accuracy'] = trimmed_mean_accuracy
    #results['eval_trimmed_mean_loss'] = trimmed_mean_loss
    #results['avg_train_trimmed_mean_time'] = trimmed_mean_time
#    results['avg_train_hist_acc'] = avg_hist_acc
#    results['avg_train_hist_acc_val'] = avg_hist_acc_val

#    with open('/storage/results.pkl', 'wb') as fp:
#    	pickle.dump(results, fp)
    #print(test_accuracies)
    #print(test_losses)
    #print(test_time)

    #print('Test history avg, acc, val_acc')
    #print(avg_hist_acc)
    #print(avg_hist_acc_val)

    #print('Test accuracies avg, acc, loss')
    #print(test_accuracies)
    #print(test_losses)

    #print(hist.history.keys())

    #print(trimmed_mean_accuracy)
    #print(trimmed_mean_loss)
    #print(trimmed_mean_time)

if __name__ == "__main__":
    main()


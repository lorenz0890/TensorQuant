import tensorflow as tf

def AlexNet():
    # TensorQuant is sensitive to the exact identifiers.
    # It is advised to use the full name ('tf.keras.layers.SomeLayer') or use aliases like shown here.
    Convolution2D = tf.keras.layers.Convolution2D
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    Flatten = tf.keras.layers.Flatten
    Dense = tf.keras.layers.Dense

    model = tf.keras.models.Sequential()

    with tf.name_scope("AlexNet"):
        with tf.name_scope("Convolution_Block"):
            # Add the first convolution layer
            model.add(Convolution2D(
                filters = 96,
                kernel_size = (11, 11),
                strides = (4, 4),
                padding = "valid",
                input_shape = (224, 224, 3),
                activation = "relu",
                name = "Conv1"))

            # Add a pooling layer
            model.add(MaxPooling2D(
                pool_size = (3, 3),
                padding = "valid",
                strides = (2, 2),
                name = "MaxPool1"))

            # Add the second convolution layer
            model.add(Convolution2D(
                filters = 256,
                kernel_size = (5, 5),
                strides=(1, 1),
                padding = "same",
                activation = "relu",
                name="Conv2"))

            # Add a second pooling layer
            model.add(MaxPooling2D(
                pool_size = (3, 3),
                padding = "valid",
                strides = (2, 2),
                name="MaxPool2"))

            # Add a third Convolution Layer
            model.add(Convolution2D(
                filters = 384,
                kernel_size = (3, 3),
                strides=(1, 1),
                padding = "same",
                activation = "relu",
                name="Conv3"))

            # Add a fourth Convolution Layer
            model.add(Convolution2D(
                filters = 384,
                kernel_size = (3, 3),
                strides=(1, 1),
                padding = "same",
                activation = "relu",
                name = "Conv4"))

            # Add a fifth Convolution Layer
            model.add(Convolution2D(
                filters = 256,
                kernel_size = (3, 3),
                strides=(1, 1),
                padding = "same",
                activation = "relu",
                name = "Conv5"))

            # Add a third pooling layer
            model.add(MaxPooling2D(
                pool_size = (3, 3),
                padding = "valid",
                strides = (2, 2),
                name = "MaxPool3"))


        # Flatten the network
        model.add(Flatten())

        with tf.name_scope("Dense_Block"):
            # Add a fully-connected hidden layer
            model.add(Dense(4096,
                activation="relu",
                name="Dense1"))

            # Add a fully-connected hidden layer
            model.add(Dense(4096,
                activation="relu",
                name="Dense2"))

            # Add a fully-connected output layer
            model.add(Dense(1000,
                activation="softmax",
                name="Dense3"))
    return model


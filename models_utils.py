import tensorflow as tf


def cnn_trad_fpool3(input_shape):

    model = tf.keras.Sequential(name='cnn_trad_fpool3')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=input_shape+tuple([1])))

    model.add(tf.keras.layers.Conv2D(64, (20, 8), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((1, 3)))

    model.add(tf.keras.layers.Conv2D(64, (10, 4), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((1, 1)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    return model


def cnn_one_fpool3(input_shape):

    model = tf.keras.Sequential(name='cnn_one_fpool3')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=input_shape+tuple([1])))

    model.add(tf.keras.layers.Conv2D(54, (32, 8), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((1, 3)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    return model


def smallCnnModel(input_shape):

    model = tf.keras.models.Sequential(name='smallCnnModel')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=input_shape+tuple([1])))

    model.add(tf.keras.layers.Convolution2D(32, (1, 10), padding='same', activation='relu'))
    model.add(tf.keras.layers.Convolution2D(64, (1, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1, 4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Convolution2D(64, (1, 10), padding='same', activation='relu'))
    model.add(tf.keras.layers.Convolution2D(128, (10, 1), padding='same', activation='relu'))

    model.add(tf.keras.layers.GlobalMaxPooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    return model
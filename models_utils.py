import tensorflow as tf
import types


def cnn_trad_fpool3(input_shape):

    model = tf.keras.Sequential(name='cnn_trad_fpool3')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

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

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

    model.add(tf.keras.layers.Conv2D(54, (32, 8), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((1, 3)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    return model


def cnn_one_fstride4(input_shape):

    model = tf.keras.Sequential(name='cnn_one_fstride4')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

    model.add(tf.keras.layers.Conv2D(186, (32, 8), strides=(1, 4), padding='same', activation='relu'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    return model


def cnn_one_fstride8(input_shape):

    model = tf.keras.Sequential(name='cnn_one_fstride8')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

    model.add(tf.keras.layers.Conv2D(336, (32, 8), strides=(1, 8), padding='same', activation='relu'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    return model


def smallCnnModel(input_shape):

    model = tf.keras.models.Sequential(name='smallCnnModel')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

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


def cnnModel(input_shape):

    model = tf.keras.models.Sequential(name='cnnModel')

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))
    model.add(tf.keras.layers.BatchNormalization())

    filters = [16, 32, 64, 128]

    for num_filters in filters:

        model.add(tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, name='features512'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(256, name='features256'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    return model


################################################
# NOTE: define all the models above this line! #
################################################


models = [f for f in globals().values() if type(f) == types.FunctionType]
models_names = [str(f).split()[1] for f in models]


def available_models():

    print('Available models:')
    for name in models_names:
        print(name)


def select_model(model_name, input_shape):

    model_index = models_names.index(model_name)
    model = models[model_index](input_shape)
    print('Selected model:', model_name)

    return model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(2 ** 10)

# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
# import sys
# sys.setrecursionlimit(2 ** 20)

from six.moves import range

from keras.datasets import cifar10
from keras.layers import Input, Dense, Layer, merge, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K


batch_size = 50  # train/test data size must be divisible by batch_size
nb_classes = 10
nb_epoch = 500
N = 18

death_mode = "lin_decay"  # or uniform
death_rate = 0.5

img_rows, img_cols = 32, 32
img_channels = 3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train -= X_train.mean(axis=0)
X_train *= 1 / X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test *= 1 / X_test.std(axis=0)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

add_tables = []

inputs = Input(shape=(img_channels, img_rows, img_cols))

net = Convolution2D(16, 3, 3, border_mode="same")(inputs)
net = BatchNormalization()(net)
net = Activation("relu")(net)


def residual_drop(x, input_shape, output_shape, strides=(1, 1)):
    global add_tables

    nb_filter = output_shape[0]
    conv = Convolution2D(nb_filter, 3, 3, subsample=strides, border_mode="same")(x)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Convolution2D(nb_filter, 3, 3, border_mode="same")(conv)
    conv = BatchNormalization()(conv)

    if strides[0] >= 2:
        x = AveragePooling2D(strides)(x)

    if (output_shape[0] - input_shape[0]) > 0:
        pad_shape = (batch_size,
                     output_shape[0] - input_shape[0],
                     output_shape[1],
                     output_shape[2])
        padding = K.ones(pad_shape)
        x = Lambda(lambda y: K.concatenate([y, padding], axis=1),
                   output_shape=output_shape)(x)

    _death_rate = K.variable(death_rate)
    scale = K.ones_like(conv) - _death_rate
    conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
                  output_shape=output_shape)(conv)

    out = merge([conv, x], mode="sum")
    out = Activation("relu")(out)

    gate = K.variable(1, dtype="uint8")
    add_tables += [{"death_rate": _death_rate, "gate": gate}]
    return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
                  output_shape=output_shape)([out, x])


for i in range(N):
    net = residual_drop(net, input_shape=(16, 32, 32), output_shape=(16, 32, 32))

net = residual_drop(
    net,
    input_shape=(16, 32, 32),
    output_shape=(32, 16, 16),
    strides=(2, 2)
)
for i in range(N - 1):
    net = residual_drop(
        net,
        input_shape=(32, 16, 16),
        output_shape=(32, 16, 16)
    )

net = residual_drop(
    net,
    input_shape=(32, 16, 16),
    output_shape=(64, 8, 8),
    strides=(2, 2)
)
for i in range(N - 1):
    net = residual_drop(
        net,
        input_shape=(64, 8, 8),
        output_shape=(64, 8, 8)
    )

pool = AveragePooling2D((8, 8))(net)
flatten = Flatten()(pool)

predictions = Dense(10, activation="softmax")(flatten)
model = Model(input=inputs, output=predictions)

sgd = SGD(lr=0.5, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy")


def open_all_gates():
    for t in add_tables:
        K.set_value(t["gate"], 1)


# setup death rate
for i, tb in enumerate(add_tables):
    if death_mode == "uniform":
        K.set_value(tb["death_rate"], death_rate)
    elif death_mode == "lin_decay":
        K.set_value(tb["death_rate"], i / len(add_tables) * death_rate)
    else:
        raise


class GatesUpdate(Callback):
    def on_batch_begin(self, batch, logs={}):
        open_all_gates()

        rands = np.random.uniform(size=len(add_tables))
        for t, rand in zip(add_tables, rands):
            if rand < K.get_value(t["death_rate"]):
                K.set_value(t["gate"], 0)

    def on_epoch_end(self, epoch, logs={}):
        open_all_gates()  # for validation


datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    horizontal_flip=True,
    vertical_flip=False)
datagen.fit(X_train)

test_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    horizontal_flip=False,
    vertical_flip=False)
test_datagen.fit(X_test)

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=test_datagen.flow(X_test, Y_test, batch_size=batch_size),
                    nb_val_samples=X_test.shape[0])

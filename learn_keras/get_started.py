import numpy as np

from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot

from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Merge

from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU

from keras.layers import RepeatVector
from keras.layers import TimeDistributed


def model_binary():
    model = Sequential()
    model.add(Dense(1, input_dim=784))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    data = np.random.random((1000, 784))
    labels = np.random.randint(2, size=(1000, 1))
    model.fit(data, labels, nb_epoch=10, batch_size=32)

    return model


def model_multiple():
    model = Sequential()
    model.add(Dense(32, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    data = np.random.random((1000, 784))
    labels = np.random.randint(10, size=(1000, 1))
    labels = to_categorical(labels, 10)
    model.fit(data, labels)

    return model


def model_merged():
    left_branch = Sequential()
    left_branch.add(Dense(32, input_dim=784))
    right_branch = Sequential()
    right_branch.add(Dense(32, input_dim=784))
    merged = Merge([left_branch, right_branch], mode='concat')

    model = Sequential()
    model.add(merged)
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    data_left = np.random.random((1000, 784))
    data_right = np.random.random((1000, 784))
    labels = np.random.randint(10, size=(1000, 1))
    labels = to_categorical(labels, 10)
    model.fit([data_left, data_right], labels, nb_epoch=10, batch_size=32)

    return model


def model_mlp():
    model = Sequential()
    model.add(Dense(64, input_dim=20, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    return model


def model_vggnet():
    model = Sequential()
    model.add(Convolution2D(
        32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd)

    return model


def model_image_caption():
    max_caption_len = 16
    vocab_size = 10000

    image_model = Sequential()
    image_model.add(Convolution2D(
        32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(32, 3, 3))
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(64, 3, 3))
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Flatten())
    image_model.add(Dense(128))

    # image_model.load_weights('weight_file.h5')

    language_model = Sequential()
    language_model.add(
        Embedding(vocab_size, 256, input_length=max_caption_len))
    language_model.add(GRU(output_dim=128, return_sequences=True))
    language_model.add(TimeDistributed(Dense(128)))

    image_model.add(RepeatVector(max_caption_len))

    model = Sequential()
    model.add(Merge(
        [image_model, language_model], mode='concat', concat_axis=-1))
    model.add(GRU(256, return_sequences=False))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop')

    # model.fit(
    #     [images, partial_captions], next_words, batch_size=16, nb_epoch=100)

    return model


def model_lstm():
    max_len = 10
    max_features = 10000

    model = Sequential()
    model.add(Embedding(max_features, 256, input_length=max_len))
    model.add(LSTM(
        output_dim=128, activation='sigmoid',
        inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    return model


def model_lstm_stacked():
    data_dim = 16
    timesteps = 8
    nb_classes = 10

    model = Sequential()
    model.add(LSTM(
        32, return_sequences=True, input_shape=(timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    train_data = np.random.random((1000, timesteps, data_dim))
    train_labels = np.random.random((1000, nb_classes))
    val_data = np.random.random((100, timesteps, data_dim))
    val_labels = np.random.random((100, nb_classes))

    model.fit(
        train_data, train_labels, batch_size=64, nb_epoch=5,
        validation_data=(val_data, val_labels))

    return model


def model_lstm_stateful():
    data_dim = 16
    timesteps = 8
    nb_classes = 10
    batch_size = 32

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True, stateful=True))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    train_data = np.random.random((batch_size * 10, timesteps, data_dim))
    train_labels = np.random.random((batch_size * 10, nb_classes))
    val_data = np.random.random((batch_size * 3, timesteps, data_dim))
    val_labels = np.random.random((batch_size * 3, nb_classes))

    model.fit(
        train_data, train_labels, batch_size=batch_size, nb_epoch=5,
        validation_data=(val_data, val_labels))

    return model


def model_lstm_merged():
    data_dim = 16
    timesteps = 8
    nb_classes = 10

    encoder_left = Sequential()
    encoder_left.add(LSTM(32, input_shape=(timesteps, data_dim)))

    encoder_right = Sequential()
    encoder_right.add(LSTM(32, input_shape=(timesteps, data_dim)))

    model = Sequential()
    model.add(Merge([encoder_left, encoder_right], mode='concat'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    train_data_left = np.random.random((1000, timesteps, data_dim))
    train_data_right = np.random.random((1000, timesteps, data_dim))
    train_labels = np.random.random((1000, nb_classes))
    val_data_left = np.random.random((100, timesteps, data_dim))
    val_data_right = np.random.random((100, timesteps, data_dim))
    val_labels = np.random.random((100, nb_classes))

    model.fit(
        [train_data_left, train_data_right], train_labels,
        batch_size=64, nb_epoch=5,
        validation_data=([val_data_left, val_data_right], val_labels))

    return model


def plot_model(model, file):
    plot(model, to_file=file)


if __name__ == '__main__':
    model = model_binary()
    plot(model, 'model_binary.png')

    model = model_multiple()
    plot(model, 'model_multiple.png')

    model = model_merged()
    plot(model, 'model_merged.png')

    model = model_mlp()
    plot(model, 'model_mlp.png')

    model = model_vggnet()
    plot(model, 'model_vggnet.png')

    model = model_image_caption()
    plot(model, 'model_image_caption.png')

    model = model_lstm()
    plot(model, 'model_lstm.png')

    model = model_lstm_stacked()
    plot(model, 'model_lstm_stacked.png')

    model = model_lstm_stateful()
    plot(model, 'model_lstm_stateful.png')

    model = model_lstm_merged()
    plot(model, 'model_lstm_merged.png')

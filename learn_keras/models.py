import numpy as np

from keras.models import Model
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Activation


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


def keras_model():
    model = model_binary()
    model.summary()

    config = model.get_config()
    model = Model.from_config(config)


if __name__ == '__main__':
    keras_model()

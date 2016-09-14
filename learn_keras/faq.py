import numpy as np

from keras import backend as K

from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.models import model_from_yaml

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation

from keras.callbacks import EarlyStopping


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


def io_model():
    model = model_binary()

    model.save('results/my_model.h5')
    del model
    model = load_model('results/my_model.h5')

    json_string = model.to_json()
    yaml_string = model.to_yaml()
    model = model_from_json(json_string)
    model = model_from_yaml(yaml_string)

    model.save_weights('results/my_model_weights.h5')
    model.load_weights('results/my_model_weights.h5')

    return model


def intermediate_layer():
    model = model_binary()

    get_layer_output = K.function(
        [model.layers[0].input], [model.layers[1].output])
    data = np.random.random((1000, 784))
    out = get_layer_output([data])[0]

    return out


def intermediate_layer_api():
    inputs = Input(shape=(784,))
    encoded = Dense(32, activation='relu')(inputs)
    decoded = Dense(784)(encoded)
    model = Model(input=inputs, output=decoded)

    encoder = Model(input=inputs, output=encoded)
    data = np.random.random((1000, 784))
    out = encoder.predict(data)

    return out, model


def early_stopping():
    model = Sequential()
    model.add(Dense(1, input_dim=784))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    data = np.random.random((1000, 784))
    labels = np.random.randint(2, size=(1000, 1))
    model.fit(
        data, labels, nb_epoch=10, batch_size=32,
        validation_split=0.2, callbacks=[early_stopping])


def training_history():
    model = Sequential()
    model.add(Dense(1, input_dim=784))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    data = np.random.random((1000, 784))
    labels = np.random.randint(2, size=(1000, 1))
    hist = model.fit(
        data, labels, nb_epoch=10, batch_size=32, validation_split=0.2)

    print(hist.history)


def freeze_layers():
    inputs = Input(shape=(784,))
    layer = Dense(1)
    layer.trainable = False
    outputs = layer(inputs)

    data = np.random.random((1000, 784))
    labels = np.random.randint(2, size=(1000, 1))

    frozen_model = Model(inputs, outputs)
    frozen_model.compile(optimizer='rmsprop', loss='mse')
    frozen_model.fit(data, labels)

    layer.trainable = True
    trainable_model = Model(inputs, outputs)
    trainable_model.compile(optimizer='rmsprop', loss='mse')
    trainable_model.fit(data, labels)


def model_stateful_rnn():
    pass


def remove_layer():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=784))
    model.add(Dense(32, activation='relu'))
    print(len(model.layers))

    model.pop()
    print(len(model.layers))


def pretrained_model():
    pass
    # from keras.applications.vgg16 import VGG16
    # from keras.applications.vgg19 import VGG19
    # from keras.applications.resnet50 import ResNet50
    # from keras.applications.inception_v3 import InceptionV3

    # model_vgg16 = VGG16(weights='imagenet', include_top=True)
    # model_vgg19 = VGG19(weights='imagenet', include_top=True)
    # model_resnet50 = ResNet50(weights='imagenet', include_top=True)
    # model_inception3 = InceptionV3(weights='imagenet', include_top=True)

    # print model_vgg16
    # print model_vgg19
    # print model_resnet50
    # print model_inception3


if __name__ == '__main__':
    io_model()
    intermediate_layer()
    intermediate_layer_api()
    early_stopping()
    training_history()
    freeze_layers()
    model_stateful_rnn()
    remove_layer()
    pretrained_model()

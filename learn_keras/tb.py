import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Input, InputLayer
from keras.callbacks import TensorBoard

import keras.backend as K


def dump(model, dir):
    cb = TensorBoard(dir, write_graph=True)
    cb._set_model(model)
    cb.writer.flush()

model = Sequential()
model.add(InputLayer(input_shape=(12,)))
model.add(Dense(3))
model.add(Dense(1))

model.compile("sgd", "mse")

dump(model, "./seq-simple")


K.clear_session()

model = Sequential()

with tf.name_scope("input"):
    model.add(InputLayer(input_shape=(12,)))
with tf.name_scope("inner"):
    model.add(Dense(3))
with tf.name_scope("output"):
    model.add(Dense(1))
with tf.name_scope("loss"):
    model.compile("sgd", "mse")

dump(model, "./seq-named")

K.clear_session()

inputs = Input(shape=(12,))
inner = Dense(3)(inputs)
outputs = Dense(1)(inner)

model = Model(input=inputs, output=outputs)
model.compile("sgd", "mse")
dump(model, "./func-simple")

K.clear_session()

with tf.name_scope("input"):
    inputs = Input(shape=(12,))

with tf.name_scope("inner"):
    inner = Dense(3)(inputs)

with tf.name_scope("output"):
    outputs = Dense(1)(inner)

model = Model(input=inputs, output=outputs)

with tf.name_scope("loss"):
    model.compile("sgd", "mse")

dump(model, "./func-named")

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
from keras.models import Sequential
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


def digit_classification():
    sess = tf.Session()
    K.set_session(sess)

    img = tf.placeholder(tf.float32, shape=(None, 784))
    labels = tf.placeholder(tf.float32, shape=(None, 10))

    x = Dense(128, activation='relu')(img)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    preds = Dense(10, activation='softmax')(x)
    loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    with sess.as_default():
        for i in range(100):
            batch = mnist_data.train.next_batch(50)
            train_step.run(feed_dict={
                img: batch[0], labels: batch[1], K.learning_phase(): 1})

    acc_value = accuracy(labels, preds)
    with sess.as_default():
        print acc_value.eval(feed_dict={
            img: mnist_data.test.images,
            labels: mnist_data.test.labels,
            K.learning_phase(): 0})


def scope_name():
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    with tf.name_scope('block1'):
        y = LSTM(32, name='mylstm')(x)
    return y


def scope_device():
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, shape=(None, 20, 64))
        y = LSTM(32)(x)
    return y


def scope_graph():
    my_graph = tf.Graph()
    with my_graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 20, 64))
        y = LSTM(32)(x)
    return y


def scope_variable():
    lstm = LSTM(32)
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = tf.placeholder(tf.float32, shape=(None, 20, 64))
    x_encoded = lstm(x)
    y_encoded = lstm(y)
    return x_encoded, y_encoded


def update_states():
    pass
    # from keras.layers import BatchNormalization
    # x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    # layer = BatchNormalization()(x)
    # update_ops = []
    # for old_value, new_value in layer.updates:
    #     update_ops.append(tf.assign(old_value, new_value))


def collect_variables():
    layer = Dense(32)
    print layer.trainable_weights


def convert_model_to_tensor():
    img = tf.placeholder(tf.float32, shape=(None, 784))

    model = Sequential()
    first_layer = Dense(32, activation='relu', input_dim=784)
    first_layer.set_input(img)
    model.add(first_layer)
    model.add(Dense(10, activation='softmax'))

    output = model.output
    print type(output)


def call_model_on_tensor():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=784))
    model.add(Dense(10, activation='softmax'))

    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = model(x)
    print type(y)


def distribute_gpus():
    with tf.device('/cpu:0'):
        x = tf.placeholder(tf.float32, shape=(None, 784))
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=784))
        model.add(Dense(10, activation='softmax'))

    with tf.device('/gpu:0'):
        output_0 = model(x)

    with tf.device('/gpu:1'):
        output_1 = model(x)

    with tf.device('/cpu:0'):
        preds = 0.5 * (output_0 + output_1)
    print type(preds)

    # data = tf.random_normal([1000, 784])
    # sess = tf.Session()
    # output = sess.run([preds], feed_dict={x: data})
    # print output


def distribute_training():
    server = tf.train.Server.create_local_server()
    sess = tf.Session(server.target)
    K.set_session(sess)


if __name__ == '__main__':
    # digit_classification()
    # scope_name()
    # scope_device()
    # scope_graph()
    # scope_variable()
    # update_states()
    # collect_variables()
    # convert_model_to_tensor()
    # call_model_on_tensor()
    distribute_gpus()
    distribute_training()

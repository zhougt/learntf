from keras.models import Model
from keras.models import Sequential

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

from keras.layers import merge
from keras.layers import Merge
from keras.layers import TimeDistributed

from keras.utils.visualize_util import model_to_dot


def callable_model():
    inputs = Input(shape=(784,))

    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    x = Input(shape=(784,))
    y = model(x)

    input_sequences = Input(shape=(20, 784))
    output_sequences = TimeDistributed(model)(input_sequences)

    return y, output_sequences


def layer_outputs():
    input_a = Input(shape=(140, 256))
    input_b = Input(shape=(140, 256))
    lstm = LSTM(32)

    encoded_a = lstm(input_a)
    assert lstm.output == encoded_a

    encoded_b = lstm(input_b)
    assert lstm.get_output_at(0) == encoded_a
    assert lstm.get_output_at(1) == encoded_b


def layer_inputs():
    input_a = Input(shape=(3, 32, 32))
    input_b = Input(shape=(3, 64, 64))

    conv = Convolution2D(16, 3, 3, border_mode='same')
    conved_a = conv(input_a)
    assert conv.input_shape == (None, 3, 32, 32)

    conved_b = conv(input_b)
    assert conv.get_input_shape_at(0) == (None, 3, 32, 32)
    assert conv.get_input_shape_at(1) == (None, 3, 64, 64)


def model_multiple_ios():
    input_main = Input(shape=(100,), dtype='int32', name='input_main')

    x = Embedding(
        output_dim=512, input_dim=10000, input_length=100)(input_main)
    output_lstm = LSTM(32)(x)

    output_aux = Dense(
        1, activation='sigmoid', name='output_aux')(output_lstm)

    input_aux = Input(shape=(5,), name='input_aux')
    x = merge([output_lstm, input_aux], mode='concat')
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    output_main = Dense(1, activation='sigmoid', name='output_main')(x)

    model = Model(
        input=[input_main, input_aux],
        output=[output_main, output_aux])
    model.compile(
        optimizer='rmsprop',
        loss={
            'output_main': 'binary_crossentropy',
            'output_aux': 'binary_crossentropy'},
        loss_weights={'output_main': 1., 'output_aux': 0.2})

    # model.fit(
    #     {'input_main': headline_data, 'input_aux': additional_data},
    #     {'output_main': labels, 'output_aux': labels},
    #     nb_epoch=50, batch_size=32)

    return model


def model_shared():
    input_a = Input(shape=(140, 256))
    input_b = Input(shape=(140, 256))

    shared_lstm = LSTM(64)
    encoded_a = shared_lstm(input_a)
    encoded_b = shared_lstm(input_b)

    merged = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(input=[input_a, input_b], output=output)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # model.fit([data_a, data_b], labels, nb_epoch=10)

    return model


def model_inception():
    input_image = Input(shape=(3, 256, 256))

    tower_1 = Convolution2D(
        64, 1, 1, border_mode='same', activation='relu')(input_image)
    tower_1 = Convolution2D(
        64, 3, 3, border_mode='same', activation='relu')(tower_1)

    tower_2 = Convolution2D(
        64, 1, 1, border_mode='same', activation='relu')(input_image)
    tower_2 = Convolution2D(
        64, 5, 5, border_mode='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D(
        (3, 3), strides=(1, 1), border_mode='same')(input_image)
    tower_3 = Convolution2D(
        64, 1, 1, border_mode='same', activation='relu')(tower_3)

    output = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)

    model = Model(input=input_image, output=output)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def model_residual():
    input_image = Input(shape=(3, 256, 256))
    conv_img = Convolution2D(3, 3, 3, border_mode='same')(input_image)
    output = merge([input_image, conv_img], mode='sum')

    model = Model(input=input_image, output=output)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def model_shared_vision():
    input_digit = Input(shape=(1, 27, 27))
    x = Convolution2D(64, 3, 3)(input_digit)
    x = Convolution2D(64, 3, 3)(x)
    x = MaxPooling2D((2, 2))(x)

    output = Flatten()(x)
    vision_model = Model(input_digit, output)

    input_digit_a = Input(shape=(1, 27, 27))
    input_digit_b = Input(shape=(1, 27, 27))
    output_a = vision_model(input_digit_a)
    output_b = vision_model(input_digit_b)

    output_concatenated = merge([output_a, output_b], mode='concat')
    output = Dense(1, activation='sigmoid')(output_concatenated)

    model = Model([input_digit_a, input_digit_b], output_concatenated)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


def model_qa_image():
    vision_model = Sequential()
    vision_model.add(Convolution2D(
        64, 3, 3, activation='relu', border_mode='same',
        input_shape=(3, 224, 224)))
    vision_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Convolution2D(
        128, 3, 3, activation='relu', border_mode='same'))
    vision_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Convolution2D(
        256, 3, 3, activation='relu', border_mode='same'))
    vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    input_image = Input(shape=(3, 224, 224))
    encoded_image = vision_model(input_image)

    input_question = Input(shape=(100,), dtype='int32')
    embedded_question = Embedding(
        input_dim=10000, output_dim=256, input_length=100)(input_question)
    encoded_question = LSTM(256)(embedded_question)

    merged = merge([encoded_image, encoded_question], mode='concat')
    output = Dense(1000, activation='softmax')(merged)

    model = Model(input=[input_image, input_question], output=output)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def model_qa_image_api():
    input_image = Input(shape=(3, 224, 224))

    x = Convolution2D(
        64, 3, 3, activation='relu', border_mode='same')(input_image)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    encoded_image = Flatten()(x)

    input_question = Input(shape=(100,), dtype='int32')
    embedded_question = Embedding(
        input_dim=10000, output_dim=256, input_length=100)(input_question)
    encoded_question = LSTM(256)(embedded_question)

    merged = merge([encoded_image, encoded_question], mode='concat')
    output = Dense(1000, activation='softmax')(merged)

    model = Model(input=[input_image, input_question], output=output)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def model_qa_image_nonapi():
    vision_model = Sequential()
    vision_model.add(Convolution2D(
        64, 3, 3, activation='relu', border_mode='same',
        input_shape=(3, 224, 224)))
    vision_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Convolution2D(
        128, 3, 3, activation='relu', border_mode='same'))
    vision_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Convolution2D(
        256, 3, 3, activation='relu', border_mode='same'))
    vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    question_model = Sequential()
    question_model.add(Embedding(10000, 256, input_length=100))
    question_model.add(LSTM(256))

    model = Sequential()
    model.add(Merge([vision_model, question_model], mode='concat'))
    model.add(Dense(1000, activation='softmax'))

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def model_qa_video():
    vision_model = Sequential()
    vision_model.add(Convolution2D(
        64, 3, 3, activation='relu', border_mode='same',
        input_shape=(3, 224, 224)))
    vision_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Convolution2D(
        128, 3, 3, activation='relu', border_mode='same'))
    vision_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Convolution2D(
        256, 3, 3, activation='relu', border_mode='same'))
    vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    input_video = Input(shape=(100, 3, 224, 224))
    encoded_frames = TimeDistributed(vision_model)(input_video)
    encoded_video = LSTM(256)(encoded_frames)

    input_question = Input(shape=(100,), dtype='int32')
    embedded_question = Embedding(
        input_dim=10000, output_dim=256, input_length=100)(input_question)
    encoded_question = LSTM(256)(embedded_question)

    merged = merge([encoded_video, encoded_question], mode='concat')
    output = Dense(1000, activation='softmax')(merged)

    model = Model(input=[input_video, input_question], output=output)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def viz_model(model, file):
    dot = model_to_dot(model, show_shapes=True, show_layer_names=False)
    dot.write_pdf(file)


if __name__ == '__main__':
    callable_model()
    layer_outputs()
    layer_inputs()

    model = model_multiple_ios()
    viz_model(model, 'results/model_multiple_ios.pdf')

    model = model_shared()
    viz_model(model, 'results/model_shared.pdf')

    model = model_inception()
    viz_model(model, 'results/model_inception.pdf')

    model = model_residual()
    viz_model(model, 'results/model_residual.pdf')

    model = model_shared_vision()
    viz_model(model, 'results/model_shared_vision.pdf')

    model = model_qa_image()
    viz_model(model, 'results/model_qa_image.pdf')

    model = model_qa_image_api()
    viz_model(model, 'results/model_qa_image_api.pdf')

    model = model_qa_image_nonapi()
    viz_model(model, 'results/model_qa_image_nonapi.pdf')

    model = model_qa_video()
    viz_model(model, 'results/model_qa_video.pdf')

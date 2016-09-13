from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Merge
from keras.layers import LSTM
from keras.layers import Embedding

from keras.layers import TimeDistributed


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


def model_multiple_ios():
    main_input = Input(shape=(100,), dtype='int32', name='main_input')

    x = Embedding(
        output_dim=512, input_dim=10000, input_length=100)(main_input)
    lstm_out = LSTM(32)(x)

    auxiliary_loss = Dense(
        1, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = Input(shape=(5,), name='aux_input')
    x = Merge([lstm_out, auxiliary_input], mode='concat')

    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    main_loss = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(
        input=[main_input, auxiliary_input],
        output=[main_loss, auxiliary_loss])

    model.compile(
        optimizer='rmsprop',
        loss={
            'main_output': 'binary_crossentropy',
            'aux_output': 'binary_crossentropy'},
        loss_weights={'main_output': 1., 'aux_output': 0.2})

    # model.fit(
    #     {'main_input': headline_data, 'aux_input': additional_data},
    #     {'main_output': labels, 'aux_output': labels},
    #     nb_epoch=50, batch_size=32)


if __name__ == '__main__':
    callable_model()

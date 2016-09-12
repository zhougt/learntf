from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation


model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

print 'done!'

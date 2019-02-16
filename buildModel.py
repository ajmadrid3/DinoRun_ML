import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard

#model hyper parameters 
LEARNING_RATE = 1e-4
img_rows, img_cols = 40,20
img_channels = 4
ACTIONS = 2

def buildmodel():
    #Building keras model 
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides = (4, 4), padding = 'same', input_shape = (img_cols, img_rows, img_channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides = (2, 2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(Activation('relu'))    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr = LEARNING_RATE)
    model.compile(loss = 'mse', optimizers = adam)
    return model


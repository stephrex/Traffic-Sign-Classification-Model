import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D


class Traffic_Sign_CNN:
    @staticmethod
    def build(width, height, depth, classes):
        # instantiate variables
        model = Sequential()
        input_shape = (width, height, depth)
        # Add Conv2D => Relu => BN => MaxPool2D layers
        model.add(Conv2D(8, (2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add another set of layers
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add another set of layers
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add Fully Connected Dense Layers
        model.add(GlobalAveragePooling2D())
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization(axis=-1))
        # model.add(Dropout(0.5))
        # Add Another Set of Fully Connected Dense Layers
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.5))
        # Softmax Classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

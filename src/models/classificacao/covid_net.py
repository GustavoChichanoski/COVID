from tensorflow.python import keras
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers.core import Activation, Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization

class CovidNet(keras.Model):

    def __init__(
        self
    ) -> None:

        self.conv = [None] * 4
        self.bn = [BatchNormalization() for _ in range(4)]
        self.relu = [Activation('relu') for _ in range(4)]
        self.max = [MaxPooling2D() for _ in range(2)]

        self.flatten = Flatten()
        self.drop = Dropout(rate=0.5)
        self.act_1 = Activation('sigmoid')
        self.act_2 = Activation('softmax')

        self.dense = [Dense(128) for _ in range(2)]

    def _covid_conv(
        self,
        k: int,
        filters: int
    ) -> Layer:
        self.conv[k] = Conv2D(
            filters=filters,
            kernel_size=(3,3),
            padding='same'
        )
        self.bn[k] = BatchNormalization()
        self.relu[k] = Activation('relu')

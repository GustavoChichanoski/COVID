from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Conv2D
import tensorflow_addons as tfa
import pandas as pd
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2

def severity_model(
    input_shape: Tuple[int,int,int] = (224,224,1),
    degrees: int = 3,
) -> Model:
    """ Função de criação do modelo de classificação da severidade da Covid-19.

        Args:
            input_shape: (Tuple[int,int,int], optional): define the input shape of model. Default to (224,224,1)
            degrees (int, optional): Define o numero de classes de saida. Defaults to 3 (Mild, Moderate, Severe).

        Returns:
            Model: Modelo de aprendizado da rede neural.
    """
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Conv2D(3,(3,3),padding='same',activation='relu'))
    model.add(ResNet50V2(include_top=False,pooling='avg',input_shape=(224,224,3)))
    model.add(Flatten())
    model.add(Dense(1,activation='relu'))
    return model


def compile_model(model: Model) -> None:
    model.compile(loss='sparse_categorical_crossentropy')
    return None
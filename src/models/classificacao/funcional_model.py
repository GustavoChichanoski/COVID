from typing import List, Tuple, Union
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2


def classification_model(
    dim: int = 256,
    channels: int = 1,
    classes: int = 3,
    final_activation: str = "softmax",
    activation: str = "relu",
    drop_rate: float = 0.2,
) -> Model:
    """Função de criação do modelo de classificao da doença presente no pulmão.

    Args:
        dim (int, optional): Dimensão da imagem de entrada `[0...]`. Defaults to `256`.
        channels (int, optional): Numero de canais da imagem de entrada `[0...3]`. Defaults to `1`.
        classes (int, optional): Numero de classes `[0...]`. Defaults to `3`.
        final_activation (str, optional): Tipo de ativação da ultima camada Dense. Defaults to `'softmax'`.
        activation (str, optional): Tipo de ativação da convolução. Defaults to `'relu'`.
        drop_rate (float, optional): Taxa de Dropout. Defaults to `0.2`.

    Returns:
        Model: [description]
    """
    input_shape = (dim, dim, channels)
    inputs = Input(shape=input_shape)
    layers = BatchNormalization()(inputs)
    layers = Conv2D(filters=3, kernel_size=(3, 3))(layers)
    layers = Activation(activation=activation)(layers)
    layers = Dropout(rate=drop_rate)(layers)
    layers = ResNet50V2(include_top=False, input_shape=(dim, dim, 3))(layers)
    layers = Flatten()(layers)
    layers = Dense(units=classes)(layers)
    layers = Activation(activation=final_activation)(layers)
    outputs = Dropout(rate=drop_rate)(layers)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def names_classification_layers(model: Model) -> List[str]:
    layers_names = []
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            if isinstance(layer, Conv2D):
                break
            layers_names.insert(0,layer.name)
        else:
            if isinstance(layer, Conv2D):
                break
            layers_names.insert(0,layer.name)
    return layers_names


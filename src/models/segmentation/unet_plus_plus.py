from typing import List, Tuple
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D


def conv_plus_plus(
    layer: Layer,
    filters: int = 32,
    kernel_size: int = 3,
    activation: str = "relu",
    matrix_position: Tuple[int, int] = (0, 0)
) -> Layer:

    cv_name = f"cv_{matrix_position[0]}_{matrix_position[1]}"
    bn_name = f"bn_{matrix_position[0]}_{matrix_position[1]}"
    act_name = f"act_{matrix_position[0]}_{matrix_position[1]}"

    kernel = (kernel_size, kernel_size)

    layer = Conv2D(filters, kernel_size=kernel, padding="same", name=cv_name)(layer)
    layer = BatchNormalization(name=bn_name)(layer)
    layer = Activation(activation=activation, name=act_name)(layer)

    return layer


def backbone(
    layer: Layer,
    depth: int = 5,
    filters: int = 32,
    kernel_size: int = 3,
    activation: str = "relu",
) -> Layer:
    for x in range(depth):
        matrix_position = (0, x)
        filters = filters * 2 ** x
        layer = conv_plus_plus(layer, filters, kernel_size, activation, matrix_position)
        layer = MaxPooling2D(padding="same", name=f"max_{x}")(layer)
    return layer


def double_values_n_times(
    initial_value: int = 32,
    array_size: int = 5,
) -> List[int]:
    """Function to generate a list of double values values of initial value passed.

    Ex:
        >>> doubles_values = double_values_n_times(5,3)
        >>> print(doubles_values)
        <<< [5,10,20]

    Args:
        initial_value (int, optional): initial value of array. Defaults to 32.
        array_size (int, optional): Size of output array. Defaults to 5.

    Returns:
        List[int]: array with double values.
    """
    double_values = [initial_value]
    if array_size > 1:
        for _ in range(array_size - 1):
            double_values.append(double_values[-1] * 2)
    return double_values


def unet_plus_plus(
    depth: int = 5,
    dim: int = 256,
    channels: int = 1,
    filters: int = 32,
    final_activation: str = "sigmoid",
    activation: str = "relu",
) -> Model:
    input_shape = (dim, dim, channels)
    inputs = Input(input_shape)

    layers = {}

    filters_values = double_values_n_times(filters, depth)

    for x in range(depth):
        for y in range(depth):
            params = {'activation':activation, 'matrix_position':(x,y)}
            if x == 0:
                if y == 0:
                    layer = conv_plus_plus(layer=inputs,filters=filters_values[depth-1-y],**params)
                    layers[f'neuron_{y}_{x}'] = layer
                else:
                    layers[f'neuron_{y}_{x}'] = conv_plus_plus(
                        layer=layers[f'down_{y-1}_{x}'],
                        filters=filters_values[depth-1-y],
                        **params
                    )
                if y > 0:
                    layers[f'up_{y}_{x}'] = UpSampling2D()(layers[f'neuron_{y}_{x}'])
                if y < depth - 1:
                    layers[f'down_{y}_{x}'] = MaxPooling2D(padding='same')(layers[f'neuron_{y}_{x}'])
            else:
                if x < depth - y:
                    concat_layers = [layers[f'neuron_{y}_{k}'] for k in range(x)]
                    concat_layers.append(layers[f'up_{y+1}_{x-1}'])
                    layers[f'concat_{y}_{x}'] = Concatenate()(concat_layers)
                    layers[f'neuron_{y}_{x}'] = conv_plus_plus(
                        layer=layers[f'concat_{y}_{x}'],
                        filters=filters_values[depth-1-y],
                        **params
                    )
                    if y > 0:
                        layers[f'up_{y}_{x}'] = UpSampling2D()(layers[f'neuron_{y}_{x}'])

    outputs = conv_plus_plus(
        layer=layers[f'neuron_{0}_{depth-1}'],
        filters=1,
        kernel_size=1,
        activation=final_activation,
        matrix_position=(0,depth)
    )

    model = Model(inputs=inputs, outputs=outputs)
    return model
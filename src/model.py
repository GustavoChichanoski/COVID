"""
Biblioteca contendo as informações referente ao modelo.
"""
from keras.applications import ResNet50V2
from keras.layers import Dense
from keras import Sequential

SPLIT_SIZE = 224
SPLIT_CHANNEL = 3
N_CLASS = 3


def model_classification(input_size=(SPLIT_SIZE, SPLIT_SIZE, SPLIT_CHANNEL), n_class=N_CLASS):
    """Modelo de classificação entre covid, normal e pneumonia

    Args:
        input_size (tuple, optional): Tamanho da imagem de ebtrada. Defaults to (224, 224, 3).
        n_class (int, optional): Número de classes de saída. Defaults to 3.

    Returns:
        (keras.Model) : Modelo do keras
    """
    resnet = ResNet50V2(include_top=False,
                        weights="imagenet",
                        input_shape=input_size,
                        pooling='avg')
    resnet.trainable = False
    output = Sequential([resnet, Dense(n_class, activation='softmax')])
    return output

from pathlib import Path
from re import split
from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras.layers.convolutional import Conv
from src.plots.plots import plot_gradcam
from src.models.grad_cam_split import prob_grad_cam
from src.images.read_image import read_images
from src.dataset.classification.cla_generator import ClassificationDatasetGenerator
from src.output_result.folders import pandas2csv
from src.prints.prints import print_info
from typing import List, Union
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Activation
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.callbacks import (
    Callback,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3
import tensorflow_addons as tfa
import numpy as np


def get_callbacks() -> List[Callback]:
    """
    Retorna a lista callbacks do modelo
    Args:
    -----
        weight_path: Caminho para salvar os checkpoints
    Returns:
    --------
        (list of keras.callbacks): lista dos callbacks
    """
    # Salva os pesos dos modelo para serem carregados
    # caso o monitor não diminua
    check_params = {
        "monitor": "val_loss",
        "verbose": 1,
        "mode": "min",
        "save_best_only": True,
        "save_weights_only": True,
    }
    checkpoint = ModelCheckpoint("./model/best.weights.hdf5", **check_params)

    # Reduz o valor de LR caso o monitor nao diminuia
    reduce_params = {
        "factor": 0.5,
        "patience": 3,
        "verbose": 1,
        "mode": "max",
        "min_delta": 1e-3,
        "cooldown": 2,
        "min_lr": 1e-8,
    }
    reduce_lr = ReduceLROnPlateau(monitor="val_f1", **reduce_params)

    # Termina se um peso for NaN (not a number)
    terminate = TerminateOnNaN()
    callbacks = [checkpoint, reduce_lr, terminate]
    return callbacks


def base(model_name: str = "ResNet50V2", split_dim: int = 224) -> Model:
    """Retorna a função intermediaria para a rede utilizada.

    Args:
        model_name (str, optional): Nome do modelo intermediario. Defaults to "ResNet50V2".
        split_dim (int, optional): tamaho da imagem de entrada. Defaults to 224.

    Returns:
        Model: Modelo intermediario de aprendizado.
    """
    base = None
    shape = (split_dim, split_dim, 3)
    params = {
        "include_top": False,
        "weights": "imagenet",
        "pooling": "avg",
        "input_shape": shape,
    }
    if model_name == "VGG19":
        base = VGG19(**params)
    elif model_name == "InceptionResNetV2":
        base = InceptionResNetV2(**params)
    elif model_name == "ResNet50V2":
        base = ResNet50V2(**params)
    elif model_name == "DenseNet121":
        base = DenseNet121(**params)
    else:
        base = MobileNetV3(**params)
    return base


def classification_model(
    dim: int = 256,
    channels: int = 1,
    classes: int = 3,
    final_activation: str = "softmax",
    activation: str = "relu",
    drop_rate: float = 0.2,
    model_name: str = "ResNet50V2",
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
        Model: MOdelo de rede neural a ser treinada pelo sistema, contendo camdas
        convolucionais e densas, cujo o meio pode alterar conforme o nomde de enetrada
    """
    input_shape = (dim, dim, channels)
    inputs = Input(shape=input_shape)
    layers = BatchNormalization()(inputs)
    layers = Conv2D(filters=3, kernel_size=(3, 3))(layers)
    layers = Activation(activation=activation)(layers)
    layers = Dropout(rate=drop_rate)(layers)
    layers = base(model_name=model_name, split_dim=224)(layers)
    layers = Flatten()(layers)
    layers = Dense(units=classes)(layers)
    layers = Activation(activation=final_activation)(layers)
    outputs = Dropout(rate=drop_rate)(layers)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def last_conv_layer(model: Model) -> str:
    """Find last conv layer in the model.

    Args:
        model (Model): model to analyse.

    Returns:
        str: name of last convolution layer.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for in_layer in reversed(layer.layers):
                if isinstance(layer, Conv):
                    return in_layer.name
        if isinstance(layer, Conv):
            return layer.name
    return ""


def names_classification_layers(model: Model) -> List[str]:
    """Search in all model for find a Conv2D layer in reversed order,
    saving all the names in the track.

    Args:
        model (Model): Mdoe

    Returns:
        List[str]: layers names of classification model.

    >>> cla_layers_names = names_classification_layer(model)
    """
    layers_names = []
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            if isinstance(layer, Conv2D):
                break
            layers_names.insert(0, layer.name)
        else:
            if isinstance(layer, Conv2D):
                break
            layers_names.insert(0, layer.name)
    return layers_names


def save_weights(
    model: Model,
    model_name: str,
    history: History = None,
    parent: Path = None,
    history_path: Path = None,
    overwrite: bool = True,
    metric: str = "val_f1",
    **params,
) -> None:
    filename = model_name
    if history is not None:
        metric_value = history.history[metric][-1]
        filename = f"{filename}_{metric}_{metric_value:0.2f}"
        if history_path is not None:
            pandas2csv(history.history, history_path)
    filename = parent / filename if parent is not None else filename
    filename = f"{filename}.hdf5"
    print_info(f"Pesos salvos em: {filename}")
    return model.save_weights(filename, overwrite=overwrite, **params)


def fit(
    model: Model,
    x: ClassificationDatasetGenerator,
    validation_data: ClassificationDatasetGenerator,
    epochs: int = 100,
    batch_size: int = 32,
    shuffle: bool = True,
    callbacks: List[Callback] = None,
    **params,
) -> History:
    callbacks = get_callbacks()
    return model.fit(
        x=x,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=callbacks,
        **params,
    )


def winner(
    labels: List[str] = ["Covid", "Normal", "Pneumonia"], votes: List[int] = [0, 0, 0]
) -> str:
    """
    Retorna o label da doenca escolhido
    Args:
    -----
        labels (list): nomes das classes
        votes (list): predicao das imagens
    Returns:
    --------
        elect (str): label escolhido pelo modelo
    """
    poll = np.sum(votes, axis=0)
    elect = labels[np.argmax(poll)]
    return elect


def predict(
    model: Model, x: ClassificationDatasetGenerator, **params
) -> tfa.types.TensorLike:
    return model.predict(x, **params)


def get_classifier_layer_names(model: Model) -> List[str]:
    classifier_layers_names = []
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for inner_layer in layer.layers:
                if isinstance(layer, Conv):
                    break
                classifier_layers_names.append(inner_layer.name)
        if isinstance(layer, Conv):
            break
        classifier_layers_names.append(inner_layer.name)
    return classifier_layers_names


def get_last_conv_layer_name(model) -> Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for inner_layer in reversed(layer.layers):
                if isinstance(layer, Conv):
                    return inner_layer.name
        if isinstance(layer, Conv):
            return layer.name


def make_grad_cam(
    model: Model,
    image: Union[str, Path],
    n_splits: int = 100,
    threshold: float = 0.35,
    verbose: bool = True,
    split_dim: int = 224,
    orig_dim: int = 1024,
    channels: int = 3,
    labels: List[str] = ["Covid", "Normal", "Pneumonia"],
) -> str:
    params_splits = {
        "verbose": verbose,
        "dim": split_dim,
        "channels": channels,
        "threshold": threshold,
        "n_splits": n_splits,
    }
    cuts, positions = split(image, **params_splits)
    shape = (n_splits, split_dim, split_dim, channels)
    cuts = cuts.reshape(shape)
    imagemColor = read_images(image, color=True)
    class_names = get_classifier_layer_names(model)
    last_conv_layer_name = get_last_conv_layer_name(model)
    heatmap = prob_grad_cam(
        cuts_images=cuts,
        classifier=class_names,
        last_conv_layer_name=last_conv_layer_name,
        paths_start_positions=positions,
        model=model,
        dim_orig=orig_dim,
        winner_pos=labels.index(image[0].parts[-2]),
    )
    plot_gradcam(heatmap, imagemColor, True)
    votes = predict(model, cuts)
    elect = winner(labels=labels, votes=votes)
    return elect
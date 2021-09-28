from pathlib import Path
from typing import List, Optional, Union
from matplotlib import cm, pyplot as plt

from tensorflow.python.eager.monitoring import Metric
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.losses import Loss

from tensorflow_addons.utils.types import Optimizer

from src.images.process_images import split
from src.dataset.classification.cla_generator import ClassificationDatasetGenerator
from src.images.read_image import read_images
from src.models.grad_cam_split import last_act_after_conv_layer, prob_grad_cam
from src.output_result.folders import pandas2csv
from src.prints.prints import print_info

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
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3

import tensorflow_addons as tfa
import numpy as np
from tqdm import tqdm

from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import img_to_array

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
    checkpoint = ModelCheckpoint(".\\.model\\best.weights.hdf5", **check_params)

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
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", **reduce_params)

    # Termina se um peso for NaN (not a number)
    terminate = TerminateOnNaN()
    callbacks = [checkpoint, reduce_lr, terminate]
    return callbacks


def model_compile(
    model: Model,
    optimizer: Optional[Union[str, Optimizer]] = None,
    loss: Union[str, Loss] = "categorical_crossentropy",
    metrics: Optional[List[Metric]] = None,
    lr: float = 1e-5,
    **kwargs,
) -> None:
    """
    Compile the model with loss and metrics define by the user.

    Args:
        optimizer (optional | Optimizer): Optimizer of model. Defaults to None.
        loss (str | Loss, optional): Loss of model. Defaults to "categorical_crossentropy".
        metrics (List[Metric], optional): Metrics of systems. Defaults to None.
        lr (float, optional): Learning rate of optimizer. Defaults to 1e-5.

    Returns:
        None: compile the model with hiperparameters
    """
    optimizer = Adamax(learning_rate=lr) if optimizer is None else optimizer
    metrics = ["accuracy"] if metrics is None else metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)


def base(model_name: str = "ResNet50V2", split_dim: int = 224) -> Model:
    """
    Retorna a função intermediaria para a rede utilizada.

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
    """
    Função de criação do modelo de classificao da doença presente no pulmão.

    Args:
        dim (int, optional): Dimensão da imagem de entrada `[0...]`. Defaults to `256`.
        channels (int, optional): Numero de canais da imagem de entrada `[0...3]`. Defaults to `1`.
        classes (int, optional): Numero de classes `[0...]`. Defaults to `3`.
        final_activation (str, optional): Tipo de ativação da ultima camada Dense. Defaults to `'softmax'`.
        activation (str, optional): Tipo de ativação da convolução. Defaults to `'relu'`.
        drop_rate (float, optional): Taxa de Dropout. Defaults to `0.2`.

    Returns:
        Model: Modelo de rede neural a ser treinada pelo sistema, contendo camdas
        convolucionais e densas, cujo o meio pode alterar conforme o nomde de enetrada
    """
    input_shape = (dim, dim, channels)
    inputs = Input(shape=input_shape)
    layers = BatchNormalization()(inputs)
    layers = Conv2D(filters=3, kernel_size=(3, 3), padding="same")(layers)
    layers = Activation(activation=activation)(layers)
    layers = Dropout(rate=drop_rate)(layers)
    layers = base(model_name=model_name, split_dim=224)(layers)
    layers = Flatten()(layers)
    layers = Dense(units=classes)(layers)
    layers = Activation(activation=final_activation)(layers)
    outputs = Dropout(rate=drop_rate)(layers)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def confusion_matrix(
    model: Model,
    x: ClassificationDatasetGenerator,
    n_splits: int = 1,
    labels: List[str] = ["Covid", "Normal", "Pneumonia"],
) -> tfa.types.TensorLike:
    """
    Metódo utilizado para avaliar o desempenho de uma rede de classificação.
    A diagonal principal contem os valores preditos corretamente, enquantos os demais
    valores são as predições incorretas realizadas pelo modelo.
    >>> modelo.confusion_matrix(teste.x,n_splits=2)
    Esse código gerará uma matriz de confução para as imagens teste.x usando
    `2` recortes por imagens, explicitado em `n_splits`.
    Args:
        x (DataGenerator):
            Gerador contendo os caminhos das imagens a serem preditas.
        n_splits (int, optional):
            Numero de recortes randomicos utilizados para gerar a predição.
            Defaults to 1.
    Returns:
        (np.array):
            [Matriz contendo os valores da matriz de confusão]
    """
    n_labels = len(labels)
    matriz = np.zeros((n_labels, n_labels))
    DIM_SPLIT = 224
    for path in tqdm(x.x):
        elect = make_grad_cam(
            model=model,
            image=path,
            n_splits=n_splits,
            verbose=False,
            split_dim=DIM_SPLIT
        )
        true_index = labels.index(path.parts[-2])
        index = labels.index(elect)
        matriz[true_index][index] += 1
    return matriz


def last_conv_layer(model: Model) -> str:
    """
    Find last conv layer in the model.

    Args:
        model (Model): model to analyse.

    Returns:
        str: name of last convolution layer.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            return last_conv_layer(layer)
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


def winner(
    labels: List[str] = ["Covid", "Normal", "Pneumonia"], votes: List[int] = [0, 0, 0]
) -> str:
    """
    Retorna o label da doença escolhido
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


def get_classifier_layer_names(model: Model, layer_name: str) -> List[str]:
    """Return list string of classifier layer names

    Args:
        model (Model): model to analyse

    Returns:
        List[str]: list of classifier layers
    """
    classifier_layers_names = []
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for inner_layer in reversed(layer.layers):
                if inner_layer.name == layer_name:
                    return classifier_layers_names
                classifier_layers_names.insert(0, inner_layer.name)
        if layer.name == layer_name:
            return classifier_layers_names
        classifier_layers_names.insert(0, layer.name)
    return classifier_layers_names


def get_last_conv_layer_name(model: Model) -> Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, Conv):
                    return inner_layer.name
        if isinstance(layer, Conv):
            return layer.name


def find_base(model: Model) -> Model:
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            return layer


def save_weights(
    modelname: str,
    model: Model,
    history: History = None,
    parent: Path = None,
    history_path: Path = Path.cwd(),
    overwrite: bool = True,
    metric: str = "val_loss",
    **params,
):
    filename = modelname
    if history is not None:
        metric_value = history.history[metric][-1]
        filename = f"{filename}_{metric}_{metric_value:0.2f}"
        if history_path is not None:
            history_path = history_path / f"{filename}"
            pandas2csv(history=history, history_path=history_path)
    filename = parent / filename if parent is not None else filename
    filename = f"{filename}.hdf5"
    print_info(f"Pesos salvos em: {filename}")
    return model.save_weights(filename, overwrite=overwrite, **params)


# def cut_and_predict(
#     model:Model,
#     n_splits: int = 100,
#     threshold: float = 0.1,
# ) -> str:


def make_grad_cam(
    model: Model,
    image: Union[str, Path],
    n_splits: int = 100,
    threshold: float = 0.35,
    verbose: bool = True,
    split_dim: int = 224,
    orig_dim: int = 1024,
    channels: int = 1,
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
    shape = (1, n_splits, split_dim, split_dim, channels)
    if isinstance(image, list):
        shape = (len(image), n_splits, split_dim, split_dim, channels)
    cuts = cuts.reshape(shape)
    imagemColor = read_images(image, color=True)
    if verbose:
        last_conv_layer_name = last_act_after_conv_layer(model).name
        class_names = get_classifier_layer_names(model, last_conv_layer_name)
        print(class_names)

        if isinstance(image, list):
            winner_label = labels.index(image[0].parts[-2])
        else:
            winner_label = labels.index(image.parts[-2])

        heatmap = prob_grad_cam(
            cuts_images=cuts,
            classifier=class_names,
            last_conv_layer_name=last_conv_layer_name,
            paths_start_positions=positions,
            model=model,
            dim_orig=orig_dim,
            winner_pos=winner_label,
        )
        plot_gradcam(heatmap, imagemColor, True)
    predict_params = {"verbose": 0}
    cuts = np.reshape(cuts, (n_splits, split_dim, split_dim, channels))
    votes = predict(model, cuts, **predict_params)
    elect = winner(labels=labels, votes=votes)
    return elect

def plot_gradcam(
    heatmap: tfa.types.TensorLike,
    image: tfa.types.TensorLike,
    grad: bool = True,
    name: str = None,
    dim: int = 1024,
    alpha = 0.4
) -> str:
    """ Plota o gradCam probabilstico recebendo como parametro o
        mapa de calor e a imagem original. Ambos de mesmo tamanho.

        Args:
        -----
            heatmap (np.array): Mapa de calor
            image (np.array): Imagem original
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_color = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_color[heatmap]

    jet_heatmap0 = array_to_img(jet_heatmap)
    jet_heatmap1 = jet_heatmap0.resize((dim, dim))
    jet_heatmap2 = img_to_array(jet_heatmap1)

    superimposed_image = jet_heatmap2 * alpha + image
    superimposed_image = array_to_img(superimposed_image)

    fig = plt.figure()
    plt.imshow(superimposed_image, cmap='gray')
    # Salvar imagem
    path = ''
    if name is not None:
        path = '{}.png'.format(name)
        plt.savefig(path,dpi=fig.dpi)
    if grad:
        plt.show()
    return path
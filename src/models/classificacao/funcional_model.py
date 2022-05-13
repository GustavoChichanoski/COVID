from pathlib import Path
from typing import List, Optional, Union
from matplotlib import cm, pyplot as plt

from tensorflow.python.eager.monitoring import Metric
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.losses import Loss

from tensorflow_addons.utils.types import Optimizer

from src.images.process_images import split
from src.data.classification.cla_generator import ClassificationDatasetGenerator
from src.images.read_image import read_images
from src.models.grad_cam_split import prob_grad_cam
from src.output_result.folders import pandas2csv
from src.prints.prints import print_info

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Activation
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.normalization.batch_normalization_v1 import BatchNormalization
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
import cv2
from tqdm import tqdm

from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import img_to_array

def get_callbacks() -> List[Callback]:
    """
    Retorna a lista callbacks do modelo

    Returns:
    --------
        (List[Callback]): lista dos callbacks
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

    >>> model = classification_model(256, 1, 3, 'softmax', 'relu', 0.2, 'ResNet50V2')

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
    layers = base(model_name=model_name, split_dim=dim)(layers)
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
            split_dim=DIM_SPLIT,
        )
        true_index = labels.index(path.parts[-2])
        index = labels.index(elect)
        matriz[index][true_index] += 1
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
        model (Model): model to analyse classification layers.

    Returns:
        List[str]: layers names of classification model.

    >>> cla_layers_names = names_classification_layer(model)
    """
    layers_names = []
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for inner in layer.layers:
                if isinstance(inner, Conv2D):
                    return layers_names
                layers_names.insert(0, inner.name)
            layers_names.insert(0, layer.name)
        else:
            if isinstance(layer, Conv2D):
                return layers_names
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
    """
    Save wights of model in `parent` folder with name of `model_name`,
    if `history` is not `None` the name of save file will be add the
    last `metric` value of model to file name. With `history_path`
    is passed to the history is save in csv file.

    Args:
        model (Model): model of weights to save.
        model_name (str): file name of weights.
        history (History, optional): history of model train. Defaults to None.
        parent (Path, optional): parent file where the files to save will be storeage. Defaults to None.
        history_path (Path, optional): path to save `history` in csv. Defaults to None.
        overwrite (bool, optional): `overwrite` file if already exists. Defaults to True.
        metric (str, optional): metrics to analyse the weights. Defaults to "val_f1".
    """
    filename = model_name
    if history is not None:
        metric_value = history.history[metric][-1]
        filename = f"{filename}_{metric}_{metric_value:0.2f}"
        if history_path is not None:
            pandas2csv(history, history_path)
    filename = parent / filename if parent is not None else filename
    filename = f"{filename}.hdf5"
    print_info(f"Pesos salvos em: {filename}")
    return model.save_weights(filename, overwrite=overwrite, **params)


def winner(
    labels: List[str] = ["Covid", "Normal", "Pneumonia"],
    votes: List[int] = [0, 0, 0]
) -> str:
    """
    Return label of disease winner.
    Args:
    -----
        labels (list): labels.
        votes (list): predictions.
    Returns:
    --------
        elect (str): winner label
    """
    poll = np.sum(votes, axis=0)
    elect = labels[np.argmax(poll)]
    return elect


def predict(
    model: Model,
    x: ClassificationDatasetGenerator,
    **params
) -> tfa.types.TensorLike:
    """Predict images from generator.

    Args:
        model (Model):
        x (ClassificationDatasetGenerator): generator who contains image to analyse.

    Returns:
        tfa.types.TensorLike: predictions.
    """
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


def get_classifier_layer(
    model: Model,
    classifier_names: List[str] = None,
) -> List[str]:

    classifier_names = [] if classifier_names is None else classifier_names
    last_activation = ''

    for layer in reversed(model.layers):

        if isinstance(layer, Model):
            return get_classifier_layer(model, classifier_names)

        if isinstance(layer, Conv):
            break

        if isinstance(layer, Activation):
            last_activation = layer.name

        classifier_names.insert(0, layer.name)

    for classifier in classifier_names:
        if classifier != last_activation:
            classifier_names.remove(classifier)
        else:
            break

    return classifier_names

def get_last_conv_layer_name(model: Model) -> Layer:
    """Get a layer in model with submodels.

    Args:
        model (Model): model to search for layer

    Returns:
        Layer: layer wanted.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, Conv):
                    return inner_layer.name
        if isinstance(layer, Conv):
            return layer.name

def find_base(model: Model) -> Model:
    """Get first submodel in a model.

    Args:
        model (Model): model to analyse.

    Returns:
        Model: first submodel in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            return layer

def predicts_values(
    model: Model,
    x: Union[str, Path],
    n_splits: int = 100,
    threshold: float = 0.35,
    verbose: bool = True,
    split_dim: int = 224,
    channels: int = 1,
) -> str:
    predicts = np.array([])
    for image in tqdm(x.x):
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
        predict_params = {"verbose": 0}
        cuts = np.reshape(cuts, (n_splits, split_dim, split_dim, channels))
        predict_array = predict(model, cuts, **predict_params)
        zeros = np.zeros((3))
        for b in predict_array:
            zeros += b
        zeros /= n_splits
        predicts = np.append(zeros, predicts)
    predicts = np.reshape(predicts, (len(x.x),3))
    return predicts

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
    image_color = read_images(image, color=True)
    if verbose:
        class_names = get_classifier_layer(model=model)

        if isinstance(image, list):
            winner_label = labels.index(image[0].parts[-2])
        else:
            winner_label = labels.index(image.parts[-2])

        heatmap = prob_grad_cam(
            cuts_images=cuts,
            classifier=class_names,
            paths_start_positions=positions,
            model=model,
            dim_orig=orig_dim,
            winner_pos=winner_label,
        )
        plot_gradcam(heatmap, image_color, True)
    predict_params = {"verbose": 0}
    cuts = np.reshape(cuts, (n_splits, split_dim, split_dim, channels))
    votes = predict(model, cuts, **predict_params)
    elect = winner(labels=labels, votes=votes)
    return elect

def superimposed_image_generate(
    heatmap: tfa.types.TensorLike,
    image: tfa.types.TensorLike,
    dim: int = 1024,
    alpha: float = 0.4,
) -> tfa.types.TensorLike:
    """Function to merge `heatmap` to `image` with original `dim` size with `alpha`
    opacity, if `grad` is `True` then plot grad cam probabilistic, if name is not
    `None` then save figure in the name file.

    Args:
        heatmap (tfa.types.TensorLike): heatmap to merge in image.
        image (tfa.types.TensorLike): image of input
        grad (bool, optional): flag to plot grad cam. Defaults to True.
        name (str, optional): name of png output file. Defaults to None.
        dim (int, optional): original dimension of image. Defaults to 1024.
        alpha (float, optional): opacity value. Defaults to 0.4.

    Returns:
        str: path where save png
    """
    if np.max(heatmap) < 1.1:
        heatmap = np.uint8(255 * heatmap)
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    jet = cm.get_cmap("jet")
    jet_color = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_color[heatmap]

    jet_heatmap0 = array_to_img(jet_heatmap)
    jet_heatmap1 = jet_heatmap0.resize((dim, dim))
    jet_heatmap2 = img_to_array(jet_heatmap1)
    superimposed_image = jet_heatmap2 * alpha + image
    return superimposed_image


def plot_gradcam(
    heatmap: tfa.types.TensorLike,
    image: tfa.types.TensorLike,
    grad: bool = False,
    name: str = None,
    dim: int = 1024,
    alpha=0.4,
) -> str:
    """Function to merge `heatmap` to `image` with original `dim` size with `alpha`
    opacity, if `grad` is `True` then plot grad cam probabilistic, if name is not
    `None` then save figure in the name file.

    Args:
        heatmap (tfa.types.TensorLike): heatmap to merge in image.
        image (tfa.types.TensorLike): image of input
        grad (bool, optional): flag to plot grad cam. Defaults to True.
        name (str, optional): name of png output file. Defaults to None.
        dim (int, optional): original dimension of image. Defaults to 1024.
        alpha (float, optional): opacity value. Defaults to 0.4.

    Returns:
        str: path where save png
    """
    superimposed_image = superimposed_image_generate(
        heatmap=heatmap,
        image=image,
        dim=dim,
        alpha=alpha
    )
    if grad:
        fig = plt.figure()
        plt.imshow(superimposed_image, cmap="gray")
        # Salvar imagem
        path = ""
        if name is not None:
            path = "{}.png".format(name)
            plt.savefig(path, dpi=fig.dpi)
        plt.show()
    return superimposed_image

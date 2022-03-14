"""
    Biblioteca contendo as informações referente ao modelo.
"""
from tensorflow.python.keras.losses import Loss
from tensorflow_addons.utils.types import Optimizer
from src.prints.prints import print_info
from typing import Any, List, Optional, Tuple, Union
from pathlib import Path

from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TerminateOnNaN
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3Small

from src.plots.plots import plot_gradcam as plt_gradcam
from src.models.metrics.f1_score import F1score
from src.models.grad_cam_split import prob_grad_cam
from src.images.process_images import split
from src.images.read_image import read_images as ri
from src.data.classification.cla_generator import ClassificationDatasetGenerator
from src.output_result.folders import pandas2csv

from pathlib import Path
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
class ModelCovid(Model):
    """[summary]"""

    def __init__(
        self,
        orig_dim: int = 1024,
        split_dim: int = 224,
        channels: int = 1,
        name: str = "Resnet50V2",
        labels: List[str] = ["Covid", "Normal", "Pneumonia"],
        trainable: bool = True,
        **kwargs,
    ) -> None:
        """
            Modelo de classificação entre covid, normal e pneumonia

            Args:
            -----
                input_size (tuple, optional): Tamanho da imagem de entrada. Defaults to (224, 224, 3).
                n_class (int, optional): Número de classes de saída. Defaults to 3.

            Returns:
            --------
                (keras.Model) : Modelo do keras
        """
        super(ModelCovid, self).__init__(name=f"Covid_{name}", **kwargs)
        # Variaveis internas
        self._lazy_base = None
        self._lazy_callbacks = None
        self._lazy_classifier_layers = None
        self.last_conv_layer_name = None
        self.split_dim = split_dim
        self.channels = channels
        self.shape = (self.split_dim,self.split_dim,self.channels)
        n_class = len(labels)

        # Parametros do modelo
        self.labels = labels
        self.trainable = trainable
        self.orig_dim = orig_dim
        self.split_dim = split_dim

        # Camadas do modelo
        self.input_layer = Input(shape=self.shape, name="entrada_modelo")
        self.conv_1 = Conv2D(
            filters=3,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="conv_gray_rgb"
        )
        self.drop_0 = Dropout(0.5, name="drop_0")
        self.dense_0 = Dense(units=256, name="dense_0")
        self.drop_1 = Dropout(0.5, name="drop_1")
        self.act_0 = Activation(activation="softmax",name="act_0")
        self.dense_1 = Dense(units=n_class,name="classifier")
        self.act_1 = Activation(activation="softmax",name="output")

        self.output_layer = self.call(self.input_layer)

    # Propriedades do modelo
    @property
    def callbacks(self) -> List[Callback]:
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
        if self._lazy_callbacks is None:
            check_params = {
                "monitor": "val_loss", "verbose": 1,
                "mode": "min", "save_best_only": True,
                "save_weights_only": True,
            }
            checkpoint = ModelCheckpoint("./model/best.weights.hdf5", **check_params)

            # Reduz o valor de LR caso o monitor nao diminuia
            reduce_params = {
                "factor": 0.5, "patience": 3,
                "verbose": 1, "mode": "max",
                "min_delta": 1e-3, "cooldown": 2,
                "min_lr": 1e-8,
            }
            reduce_lr = ReduceLROnPlateau(monitor="val_f1", **reduce_params)

            # Termina se um peso for NaN (not a number)
            terminate = TerminateOnNaN()
            self._lazy_callbacks = [checkpoint, reduce_lr, terminate]
        return self._lazy_callbacks

    @property
    def base(self) -> Model:
        if self._lazy_base is None:
            shape = (self.split_dim, self.split_dim, 3)
            params = {
                "include_top": False,
                "weights": "imagenet",
                "pooling": "avg",
                "input_shape": shape,
            }
            if self.name == "VGG19":
                self._lazy_base = VGG19(**params)
            elif self.name == "InceptionResNetV2":
                self._lazy_base = InceptionV3(**params)
            elif self.name == "MobileNetV2":
                self._lazy_base = MobileNetV3Small(**params)
            elif self.name == "DenseNet201":
                self._lazy_base = DenseNet201(**params)
            else:
                self._lazy_base = ResNet50V2(**params)
            self._lazy_base.trainable = self.trainable
        return self._lazy_base

    @property
    def classifier_layers_names(self) -> List[str]:
        if self._lazy_classifier_layers is None:
            self._lazy_classifier_layers = [
                self.drop_0.name,
                self.dense_0.name,
                self.act_0.name,
                self.drop_1.name,
                self.dense_1.name,
                self.act_1.name
            ]
            for layer in reversed(self.layers):
                if type(self.base) == type(layer):
                    self._lazy_classifier_layers.insert(0, layer.layers[-1].name)
                    self.last_conv_layer_name = layer.layers[-2].name
                    break
        return self._lazy_classifier_layers

    def call(self, inputs: Tuple[int,int,int] = (256,256,1)) -> Model:
        model = self.conv_1(inputs)
        model = self.base(model)
        model = self.drop_0(model)
        model = self.dense_0(model)
        model = self.drop_1(model)
        model = self.dense_1(model)
        return self.act_1(model)

    def save_weights(
        self,
        modelname: str,
        history: History = None,
        parent: Path = None,
        history_path: Path = None,
        overwrite: bool = True,
        metric: str = "val_f1",
        **params,
    ):
        filename = modelname
        if history is not None:
            metric_value = history.history[metric][-1]
            filename = f"{filename}_{metric}_{metric_value:0.2f}"
            if history_path is not None:
                pandas2csv(history.history, history_path)
        filename = parent / filename if parent is not None else filename
        filename = f"{filename}.hdf5"
        print_info(f"Pesos salvos em: {filename}")
        return super().save_weights(filename, overwrite=overwrite, **params)

    def load_weights(self, filepath: Union[Path,str]):
        return super().load_weights(filepath)

    def fit(
        self,
        x: ClassificationDatasetGenerator,
        validation_data: ClassificationDatasetGenerator,
        epochs: int = 100,
        batch_size: int = 32,
        shuffle: bool = True,
        callbacks: List[Callback] = None,
        **params,
    ) -> History:
        callbacks = self.callbacks if callbacks is None else callbacks
        return super().fit(
            x=x,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            shuffle=shuffle,
            callbacks=callbacks,
            **params,
        )

    def build(self, input_shape: tfa.types.TensorLike = None) -> None:
        is_tensor = tf.is_tensor(self.input_layer)
        super(ModelCovid, self).build(
            self.input_layer.shape if is_tensor else self.input_layer
        )
        self.call(self.input_layer)

    def compile(
        self,
        optimizer: Optional[Union[str,Optimizer]] = None,
        loss: Union[str,Loss] = "categorical_crossentropy",
        metrics: Optional[List[Metric]] = None,
        lr: float = 1e-5,
        **kwargs,
    ) -> None:
        """ Compile the model with loss and metrics define by the user.

            Args:
                optimizer (optional | Optimizer): Optimizer of model. Defaults to None.
                loss (str | Loss, optional): Loss of model. Defaults to "categorical_crossentropy".
                metrics (List[Metric], optional): Metrics of systems. Defaults to None.
                lr (float, optional): Learning rate of optimizer. Defaults to 1e-5.

            Returns:
                None: compile the model with hiperparameters
        """
        optimizer = Adamax(learning_rate=lr) if optimizer is None else optimizer
        metrics = ["accuracy", F1score()] if metrics is None else metrics
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs
        )

    def predict(
        self,
        x: ClassificationDatasetGenerator,
        **params
    ) -> tfa.types.TensorLike:
        return super().predict(x, **params)

    def winner(self,votes: List[int] = [0, 0, 0]) -> str:
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
        elect = self.labels[np.argmax(poll)]
        return elect

    def make_grad_cam(
        self,
        image: Union[str, Path],
        n_splits: int = 100,
        threshold: float = 0.35,
        verbose: bool = True,
    ) -> str:
        params_splits = {
            'verbose': verbose,
            'dim': self.split_dim,
            'channels': self.channels,
            'threshold': threshold,
            'n_splits': n_splits
        }
        cuts, positions = split(image, **params_splits)
        shape = (n_splits,self.split_dim,self.split_dim,self.channels)
        cuts = cuts.reshape(shape)
        imagemColor = ri(image, color=True)
        class_names = list(self.classifier_layers_names)
        heatmap = prob_grad_cam(
            cuts_images=cuts,
            classifier=class_names,
            last_conv_layer_name=self.last_conv_layer_name,
            paths_start_positions=positions,
            model=self,
            dim_orig=self.orig_dim,
            winner_pos=self.labels.index(image[0].parts[-2])
        )
        plt_gradcam(heatmap, imagemColor, True)
        votes = self.predict(cuts)
        elect = self.winner(votes=votes)
        return elect

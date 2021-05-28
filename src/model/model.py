"""
    Biblioteca contendo as informações referente ao modelo.
"""
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tensorflow.python.keras import Model
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TerminateOnNaN
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3Small

from src.plots.plots import plot_gradcam as plt_gradcam
from src.model.metrics.f1_score import F1score
from src.model.grad_cam_split import prob_grad_cam
from src.images.process_images import split_images_n_times as splits
from src.images.read_image import read_images as ri
from src.dataset.generator import DataGenerator


class ModelCovid:
    """[summary]"""

    def __init__(
        self,
        weight_path: str,
        model_input_shape: Tuple[int, int, int] = (224, 224, 3),
        batch_size: int = 32,
        model_name: str = "Resnet50V2",
        labels: List[str] = ["Covid", "Normal", "Pneumonia"],
    ) -> None:
        """
        Construtor da minha classe ModelCovid

        Args:
        -----
            weight_path (str):
                Caminho para salvar os pesos.

            model_input_shape (tuple, optional):
                Dimensão das imagens recortadas.
                Defaults to ```(224, 224, 3)```.

            batch_size (int, optional): 
                Pacotes por treinamento.
                Defaults to ```32```.

            model_name (str, optional):
                Nome do modelo de rede a ser importado, podendo variar
                entre ```'ResNet50V2'```, ```'VGG19'```, ```'DenseNet201'``` ou
                ```'InceptionNetV3'```.

            labels (list, optional):
                Rotulos de saída.
                Defaults to ```['Covid','Normal','Pneumonia']```.
        """
        self.batch_size = batch_size
        self.model_input_shape = model_input_shape
        self.model_name = model_name
        self.labels = labels
        self.model = classification(
            shape=self.model_input_shape, n_class=len(self.labels), model_net=model_name
        )
        self.weight_path = weight_path
        # Nomes das camadas até a ultima camada de convolução
        self.classifier_layers = []
        for layer in reversed(self.model.layers):
            if type(self.model) == type(layer):
                self.classifier_layers.insert(0, layer.layers[-1].name)
                self.last_conv_layer = layer.layers[-2].name
                break
            self.classifier_layers.insert(0, layer.name)

    def save(
        self,
        path: Path,
        name: str = None,
        history: Any = None,
        kaggle: bool = False,
        metric: str = "val_f1",
    ) -> Tuple[str, str, str]:
        """
            Salva os pesos do modelo em um arquivo.
            
            Args:
            -----
                history (list): Historico das metricas de apreendizagem
        """
        if name is not None:
            file = path / name
            self.model.save(f"{file}.hdf5", overwrite=True)
            self.model.save_weights(f"{file}_weights.hdf5", overwrite=True)
            print(f"Pesos salvos em {file}")
            return file + "_weights.hdf5"

        value = 100.00
        file = "model.hdf5"

        if history is not None:
            value = history[metric][-1] * 100
            history_path = f"history_{self.model_name}_{value:.02f}"
            if not kaggle:
                history_path = path / "history" / f"{history_path}"
            file_history = f"{history_path}.csv"
            hist_df = pd.DataFrame(history)
            with open(file_history, mode="w") as f:
                hist_df.to_csv(f)
            print(f"[INFO] Historico salvo em: {history_path}")

        file_name = f"{self.model_name}_{metric}_{value:.02f}"
 
        if not kaggle:
            file_model = path / f"{file_name}.hdf5"
 
        self.model.save(file, overwrite=True)
        print(f"[INFO] Modelo salvos em: {file}")
        file_weights = f"{file_name}_weights.hdf5"
 
        if not kaggle:
            file_weights = path / "weights" / file_weights
 
        self.model.save_weights(file_weights, overwrite=True)
        print(f"[INFO] Pesos salvos em: {file_weights}")

        return file_model, file_weights, file_history

    def load(self, path: Path) -> None:
        self.model.load_weights(str(path))

    def fit_generator(
        self,
        train_generator: DataGenerator,
        val_generator: DataGenerator,
        history_path: str = None,
        epochs: int = 100,
        shuffle: bool = True,
        workers: int = 1,
        batch_size: int = 32,
        **params,
    ):
        history = self.model.fit(
            x=train_generator,
            validation_data=val_generator,
            callbacks=get_callbacks(self.weight_path, history_path),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            workers=workers,
            **params,
        )
        return history

    def compile(
        self, loss: str = "categorical_crossentropy",
        lr: float = 1e-2, **params
    ) -> None:
        """Compila o modelo"""
        opt = Adamax(learning_rate=lr)
        self.model.compile(optimizer=opt, loss=loss, metrics=get_metrics(), **params)
        return None

    def predict(
        self,
        image: str,
        n_splits: int = 100,
        name: str = None,
        grad: bool = True
    ) -> str:
        """
            Realiza a predição do modelo podendo gerar o
            Grad-Cam Probabilistico da imagem caso ```grad``` seja
            ```True``` ou ```name``` seja diferente de ```None```.

            Args:

                image (str):
                    path da imagem a ser predita.

                n_splits (int, optional):
                    Numero de recortes aleatorios da imagem.
                    Defaults to 100.

                name (str, optional):
                    O nome do arquivo png do grad cam gerado.
                    Caso None não é gerado um arquivo png.
                    Defaults to None

                grad (bool, optional):
                    Mostra o grad cam probabilistico.
                    Default to True

            Returns:
            ganhador (str): rotulo predito pelo modulo
        """
        imagem = ri(image)
        cuts, positions = splits(
            image=imagem,
            n_split=n_splits,
            dim_split=self.model_input_shape[0],
            verbose=grad,
        )
        shape = (
            n_splits,
            self.model_input_shape[0],
            self.model_input_shape[1],
            self.model_input_shape[2],
        )
        cuts = cuts.reshape(shape)
        votes = self.model.predict(cuts)
        elect = winner(labels=self.labels, votes=votes)
        if grad or name is not None:
            imagemColor = ri(image, color=True)
            heatmap = prob_grad_cam(
                pacotes_da_imagem=cuts,
                classifier=self.classifier_layers,
                last_conv_layer=self.last_conv_layer,
                posicoes_iniciais_dos_pacotes=positions,
                modelo=self.model,
                winner_pos=self.labels.index(elect),
            )
            plt_gradcam(heatmap, imagemColor, grad, name)
        return elect

    def confusion_matrix(self, x: DataGenerator, n_splits: int = 1):
        """
        Metódo utilizado para avaliar o desempenho de uma rede de classificação.
        A diagonal principal contem os valores preditos corretamente, enquantos os demais
        valores são as predições incorretas realizadas pelo modelo.

        >>> modelo.confusion_matrix(teste.x,n_splits=2)

        Esse código gerará uma matriz de confução para as imagens teste.x usando
        ```2``` recortes por imagens, explicitado em ```n_splits```.

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
        n_labels = len(self.labels)
        matriz = np.zeros((n_labels, n_labels))
        for path in tqdm(x):
            elect = self.predict(image=path, n_splits=n_splits, grad=False, name=None)
            true_index = self.labels.index(path.parts[-2])
            index = self.labels.index(elect)
            matriz[true_index][index] += 1
        return matriz

    def custom_model(self, model: Model) -> None:
        self.model = model
        return None


def classification(
    shape: Tuple[int, int, int] = (224, 224, 3),
    n_class: int = 3,
    model_net: str = "Resnet50V2",
    resnet_train: bool = True,
) -> Model:
    """
    Modelo de classificação entre covid, normal e pneumonia

    Args:
    -----
        input_size (tuple, optional): Tamanho da imagem de entrada.
                                      Defaults to (224, 224, 3).
        n_class (int, optional): Número de classes de saída.
                                 Defaults to 3.

    Returns:
    --------
        (keras.Model) : Modelo do keras
    """
    inputs = Input(shape, name="entrada_modelo")
    model = Conv2D(
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv_gray_rgb",
    )(inputs)
    params = {
        "include_top": False,
        "weights": "imagenet",
        "input_shape": (shape[0], shape[1], 3),
        "pooling": "avg",
    }
    if model_net == "VGG19":
        base_model = VGG19(**params)
    elif model_net == "InceptionResNetV2":
        base_model = InceptionV3(**params)
    elif model_net == "MobileNetV2":
        base_model = MobileNetV3Small(**params)
    elif model_net == "DenseNet201":
        base_model = DenseNet201(**params)
    else:
        base_model = ResNet50V2(**params)
    base_model.trainable = resnet_train
    model = base_model(model)
    model = Dropout(0.5, name="drop_0")(model)
    model = Dense(units=256, name="dense_0")(model)
    model = Dropout(0.5, name="drop_1")(model)
    model = Dense(units=n_class, name="classifier")(model)
    predictions = Activation(activation="softmax", name="output")(model)
    return Model(inputs=inputs, outputs=predictions)


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


def get_metrics() -> List[Metric]:
    """
    Gera as metricas para o modelo
    Returns:
    --------
        (list): metricas do modelo
    """
    m = F1score()
    metrics = ["accuracy", m]
    return metrics


def get_callbacks(weight_path: str, history_path: str) -> List[Callback]:
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
    checkpoint = ModelCheckpoint(weight_path, **check_params)

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

    # Parada do treino caso o monitor nao diminua
    stop_params = {"mode": "max", "restore_best_weights": True, "patience": 40}
    early_stop = EarlyStopping(monitor="val_f1", **stop_params)
    # Termina se um peso for NaN (not a number)
    terminate = TerminateOnNaN()

    # Habilita a visualizacao no TersorBoard
    # tensorboard = TensorBoard(log_dir="./logs")

    # Armazena os dados gerados no treinamento em um CSV
    if history_path is not None:
        csv_logger = CSVLogger(history_path, append=True)
        # Vetor a ser passado na função fit
        callbacks = [checkpoint, early_stop, reduce_lr, terminate, csv_logger]
    else:
        # Vetor a ser passado na função fit
        # callbacks = [
        #     checkpoint,
        #     early_stop,
        #     reduce_lr,
        #     terminate
        # ]
        callbacks = [checkpoint, reduce_lr, terminate]
    # callbacks = [checkpoint, early_stop, reduce_lr, terminate]
    return callbacks
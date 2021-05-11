"""
    Biblioteca contendo as informações referente ao modelo.
"""
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.python.keras import Model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import TerminateOnNaN
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3Small
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from src.model.generator import DataGenerator
from src.model.grad_cam_split import prob_grad_cam
from src.images.process_images import split_images_n_times as splits
from src.images.read_image import read_images as ri
from src.plots.plots import plot_gradcam as plt_gradcam
from src.model.metrics.f1_score import F1score
from src.csv.save_csv import save_as_csv as save_csv
from pathlib import Path


class ModelCovid(Model):
    """[summary]
    """

    def __init__(
        self,
        weight_path: str,
        model_input_shape: Tuple[int,int,int] = (224, 224, 3),
        batch_size: int = 32,
        model_name: str = 'Resnet50V2',
        labels: List[str] = ['Covid', 'Normal', 'Pneumonia']
    ) -> None:
        """
            Construtor da minha classe ModelCovid

            Args:
            -----
                weight_path (str): Caminho para salvar os pesos.
                input (tuple, optional): Dimensão das imagens recortadas.
                                               Defaults to (224, 224, 3).
                batch_size (int, optional): pacotes por treinamento.
                                            Defaults to 32.
                epochs (int, optional): epocas de treinamento.
                                        Defaults to 10.
                labels (list, optional): Rotulos de saída.
                                         Defaults to ['Covid','Normal','Pneumonia'].
        """
        super(ModelCovid, self).__init__()
        self.batch_size = batch_size
        self.model_input_shape = model_input_shape
        self.model_name = model_name
        self.labels = labels
        self.model = classification(
            shape=self.model_input_shape,
            n_class=len(self.labels),
            model_net=model_name
        )
        self.weight_path = weight_path
        # Nomes das camadas até a ultima camada de convolução
        last_non_conv_layer = self.model.layers[0]
        self.classifier_layers = [last_non_conv_layer.get_layer(index=-1).name,
                                  'classifier',
                                  'output']
        self.last_conv_layer = last_non_conv_layer.get_layer(index=-2).name

    def save(self, path: Path,
             name: str = None,
             model: str = '',
             history=None,
             metric: str = 'val_f1'):
        """
            Salva os pesos do modelo em um arquivo
            Args:
                history (list): Historico das metricas de apreendizagem
        """
        # val_acc = (1 - history['val_loss'][-1])*100
        if name is not None:
            file = path / name
            self.model.save(f'{file}.hdf5', overwrite=True)
            self.model.save_weights(f'{file}_weights.hdf5', overwrite=True)
            print(f"Pesos salvos em {file}")
            return file + '_weights.hdf5'

        value = 100.00
        file = 'model.hdf5'

        if history is not None:
            value = history[metric][-1] * 100
            history_path = path / 'history' / f'history_{self.model_name}_{value:.02f}'
            file_history = f'{history_path}.csv'
            hist_df = pd.DataFrame(history)
            with open(file_history, mode='w') as f:
                hist_df.to_csv(f)

        file_name = f'{self.model_name}_{metric}_{value:.02f}'
        file_model = path / f'{file_name}.hdf5'
        self.model.save(file, overwrite=True)
        print(f"[INFO] Modelo salvos em: {file}")

        file_weights = path / 'weights' / f'{file_name}_weights.hdf5'
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
        **params
    ):
        history = self.model.fit(x=train_generator,
                                 validation_data=val_generator,
                                 callbacks=get_callbacks(
                                     self.weight_path,
                                     history_path),
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 workers=workers,
                                 **params)
        return history

    def compile(self,
                loss: str = 'categorical_crossentropy',
                lr: float = 0.01, **params) -> None:
        """Compila o modelo
        """
        opt = Adamax(learning_rate=lr)
        self.model.compile(optimizer=opt,
                           loss=loss,
                           metrics=get_metrics(),
                           **params)
        return None

    def predict(
        self,
        image: str,
        n_splits: int = 100,
        name: str = None,
        grad: bool = True
    ) -> str:
        """
            Realiza a predição do modelo

            Args:
            -----
                image (str): path da imagem a ser predita.
                n_splits (int, optional):
                    Numero de recortes aleatorios da imagem.
                    Defaults to 100.
                grad (boolean, optional):
                    Mostra o grad cam probabilistico. 
                    Default to True
            Returns:
            --------
               str: label ganhadora
        """
        imagem = ri(image)
        cuts, positions = splits(imagem, n_splits, verbose=grad)
        cuts = np.array(cuts)
        shape = (n_splits,
                 self.model_input_shape[0],
                 self.model_input_shape[1],
                 self.model_input_shape[2])
        cuts = cuts.reshape(shape)
        votes = self.model.predict(cuts)
        elect = winner(self.labels, votes)
        if grad or name is not None:
            heatmap = prob_grad_cam(pacotes_da_imagem=cuts,
                                    classifier=self.classifier_layers,
                                    last_conv_layer=self.last_conv_layer,
                                    posicoes_iniciais_dos_pacotes=positions,
                                    modelo=self.model,
                                    winner_pos=self.labels.index(elect))
            plt_gradcam(heatmap, imagem, grad, name)
        return elect

    def confusion_matrix(self, x, n_splits: int = 1):
        n_labels = len(self.labels)
        matriz = np.zeros((n_labels, n_labels))
        for path in tqdm(x):
            ytrue = None
            for label in self.labels:
                if label in str(path):
                    ytrue = label
                    break
            if ytrue is None:
                ytrue = 'Normal'
            elect = self.predict(image=path, n_splits=n_splits, grad=False)
            index = self.labels.index(elect)
            true_index = self.labels.index(ytrue)
            matriz[true_index][index] += 1
        return matriz

    def custom_model(self, model: Model) -> None:
        self.model = model
        return None


def classification(shape: Tuple[int, int, int] = (224, 224, 3),
                   n_class: int = 3,
                   model_net: str = 'Resnet50V2',
                   resnet_train: bool = True) -> Model:
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
    params = {'include_top': False,
              'weights': "imagenet",
              'input_shape': shape,
              'pooling': "avg"}
    if model_net == 'VGG19':
        resnet = VGG19(**params)
    elif model_net == 'InceptionResNetV2':
        resnet = InceptionV3(**params)
    elif model_net == 'MobileNetV2':
        resnet = MobileNetV3Small(**params)
    elif model_net == 'DenseNet201':
        resnet = DenseNet201(**params)
    else:
        resnet = ResNet50V2(**params)
    resnet.trainable = resnet_train
    output = Sequential()
    output.add(resnet)
    output.add(Dense(n_class,activation=None,name='classifier'))
    output.add(Activation('softmax', name='output'))
    return output


def winner(labels: List[str] = ["Covid", "Normal", "Pneumonia"],
           votes: List[int] = [0, 0, 0]) -> str:
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
    n_class = votes.shape[1]
    poll = np.zeros((1, n_class))
    for vote in votes:
        poll += vote
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
    metrics = ['accuracy', m]
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
        'monitor': 'val_loss', 'verbose': 1, 'mode': 'min',
        'save_best_only': True, 'save_weights_only': False
    }
    checkpoint = ModelCheckpoint(weight_path, **check_params)

    # Reduz o valor de LR caso o monitor nao diminuia
    reduce_params = {
        'factor': 0.5, 'patience': 3, 'verbose': 1,
        'mode': 'min', 'min_delta': 1e-3,
        'cooldown': 2, 'min_lr': 1e-8
    }
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', **reduce_params)

    # Parada do treino caso o monitor nao diminua
    stop_params = {'mode': 'min', 'restore_best_weights': True, 'patience': 40}
    early_stop = EarlyStopping(monitor='val_f1', **stop_params)
    # Termina se um peso for NaN (not a number)
    terminate = TerminateOnNaN()

    # Habilita a visualizacao no TersorBoard
    # tensorboard = TensorBoard(log_dir="./logs")

    # Armazena os dados gerados no treinamento em um CSV
    if history_path is not None:
        csv_logger = CSVLogger(history_path, append=True)
        # Vetor a ser passado na função fit
        callbacks = [
            checkpoint,
            early_stop,
            reduce_lr,
            terminate,
            csv_logger
        ]
    else:
        # Vetor a ser passado na função fit
        callbacks = [
            checkpoint,
            early_stop,
            reduce_lr,
            terminate
        ]
    # callbacks = [checkpoint, early_stop, reduce_lr, terminate]
    return callbacks


def conv_class(layer,
               filters: int = 32,
               kernel: tuple = (3, 3),
               act: str = "relu",
               i: int = 1) -> Any:
    # Define os nomes das layers
    conv_name, bn_name, act_name = f"CV_{i}", f"BN_{i}", f"Act_{i}"

    layer = Conv2D(filters=filters,
                   kernel_size=kernel,
                   padding='same',
                   kernel_regularizer=regularizers.l1_l2(),
                   bias_regularizer=regularizers.l1(),
                   activity_regularizer=regularizers.l1(),
                   name=conv_name)(layer)
    layer = BatchNormalization(name=bn_name)(layer)
    layer = Activation(activation=act,
                       name=act_name)(layer)
    return layer

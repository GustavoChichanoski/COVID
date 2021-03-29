"""
    Biblioteca contendo as informações referente ao modelo.
"""
import os
from src.model.generator import DataGenerator
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.ops.gen_array_ops import unique
from src.model.keract import n_
from typing import Any, List
import numpy as np
from tqdm import tqdm
from keras import Model
from keras import regularizers
from keras.layers import Input
from keras.layers import Dense
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import TerminateOnNaN
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adamax
from keras.applications import ResNet50V2
from keras.applications import VGG19
from keras.applications import DenseNet201
from keras.applications import InceptionResNetV2
from keras.applications import MobileNetV2
from src.model.grad_cam_split import prob_grad_cam
from src.images.process_images import split_images_n_times as splits
from src.dataset.dataset import Dataset
from src.images.read_image import read_images as ri
from src.plots.plots import plot_gradcam as plt_gradcam
from src.model.metrics.f1_score import F1score
from src.csv import save_csv as save_csv
from pathlib import Path


class ModelCovid(Model):
    """[summary]
    """

    def __init__(self,
                 weight_path: str,
                 model_input_shape: tuple = (224, 224, 3),
                 batch_size: int = 32,
                 epochs: int = 10,
                 model: str = 'Resnet50V2',
                 labels: List[str] = ['Covid', 'Normal', 'Pneumonia']):
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
        super(ModelCovid,self).__init__()
        self.model_input_shape = model_input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.labels = labels
        self.depth = 5
        self.filters = 16
        self.model = classification(
            shape=self.model_input_shape,
            n_class=len(self.labels),
            model_net=model
        )
        self.weight_path = weight_path
        # Nomes das camadas até a ultima camada de convolução
        self.classifier_layers = [self.model.layers[0]
                                      .get_layer(index=-1).name,
                                  'classifier',
                                  'output']
        self.last_conv_layer = self.model.layers[0].get_layer(index=-2).name

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
            self.model.save(file + '.hdf5', overwrite=True)
            self.model.save_weights(f'{file}_weights.hdf5', overwrite=True)
            print(f"Pesos salvos em {file}")
            return file + '_weights.hdf5'

        value = 0.00
        file = 'model.hdf5'

        if history is not None:
            value = history.history[metric][-1] * 100
            history_path = path / f'history_{model}_{value}'
            save_csv(value=history.history, labels=model, name=history_path)

        file = path / f'{model}_{metric}_{value:.02f}.hdf5'
        self.model.save(file, overwrite=True)

        file_weights = path / f'{model}_{metric}_{value:.02f}_weights.hdf5'
        self.model.save_weights(file_weights, overwrite=True)

        print(f"Pesos salvos em {file}")

    def load(self, path: str) -> None:
        self.model.load_weights(path)

    def fit_generator(self,
                      train_generator: DataGenerator,
                      val_generator: DataGenerator,
                      epochs: int = 100):
        history = self.model.fit(x=train_generator,
                                 validation_data=val_generator,
                                 callbacks=get_callbacks(self.weight_path),
                                 epochs=epochs,
                                 shuffle=True)
        return history

    def compile(self,
                loss: str = 'categorical_crossentropy',
                lr: float = 0.01) -> None:
        """Compila o modelo
        """
        opt = Adamax(learning_rate=lr)
        self.model.compile(optimizer=opt,
                           loss=loss,
                           metrics=get_metrics())
        self.model.summary()
        return None

    def predict(self, image: str,
                n_splits: int = 100,
                name: str = None,
                grad: bool = True) -> str:
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
        cuts = cuts.reshape((n_splits,
                             self.model_input_shape[0],
                             self.model_input_shape[1],
                             self.model_input_shape[2]))
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
            if ytrue is None:
                ytrue = 'Normal'
            elect = self.predict(image=path, n_splits=n_splits, grad=False)
            index = self.labels.index(elect)
            true_index = self.labels.index(ytrue)
            matriz[true_index][index] += 1
        return matriz

    def custom_model(self, model) -> None:
        self.model = model
        return None


def classification(shape: tuple = (224, 224, 3),
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

    resnet = ResNet50V2(**params)
    if model_net == 'VGG19':
        resnet = VGG19(**params)
    elif model_net == 'InceptionResNetV2':
        resnet = InceptionResNetV2(**params)
    elif model_net == 'MobileNetV2':
        resnet = MobileNetV2(**params)
    elif model_net == 'DenseNet201':
        resnet == DenseNet201(**params)
    resnet.trainable = resnet_train
    output = Sequential([resnet,
                         Dense(n_class,
                               activation=None,
                               name='classifier'),
                         Activation('softmax', name='output')])
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


def get_metrics():
    """
        Gera as metricas para o modelo
        Returns:
        --------
            (list): metricas do modelo
    """
    m = F1score()
    metrics = ['accuracy', m]
    return metrics


def get_callbacks(weight_path: str):
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
        'save_best_only': True, 'save_weights_only': True
    }
    checkpoint = ModelCheckpoint(weight_path, **check_params)

    # Reduz o valor de LR caso o monitor nao diminuia
    reduce_params = {
        'factor': 0.5, 'patience': 3, 'verbose': 1,
        'mode': 'min', 'epsilon': 1e-3,
        'cooldown': 2, 'min_lr': 1e-8
    }
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', **reduce_params)

    # Parada do treino caso o monitor nao diminua
    early_stop = EarlyStopping(monitor='val_f1',
                               mode='min',
                               restore_best_weights=True,
                               patience=40)

    # Termina se um peso for NaN (not a number)
    terminate = TerminateOnNaN()

    # Habilita a visualizacao no TersorBoard
    tensorboard = TensorBoard(log_dir="./logs")

    # Vetor a ser passado na função fit
    callbacks = [checkpoint, early_stop,
                 reduce_lr, terminate, tensorboard]
    return callbacks


def conv_class(layer,
               filters: int = 32,
               kernel: tuple = (3, 3),
               act: str = "relu",
               i: int = 1) -> Any:
    # Define os nomes das layers
    conv_name = "CV_{}".format(i)
    bn_name = "BN_{}".format(i)
    act_name = "Act_{}".format(i)

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

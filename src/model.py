"""
Biblioteca contendo as informações referente ao modelo.
"""
from keras.applications import ResNet50V2
from keras import Model
from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
from keras.metrics import CategoricalCrossentropy
from keras.metrics import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import TerminateOnNaN
from keras.callbacks import ReduceLROnPlateau
from dataset import Dataset

SPLIT_SIZE = 224
SPLIT_CHANNEL = 3
N_CLASS = 3


class ModelCovid:

    def __init__(self,
                 weight_path: str,
                 input_shape=(224, 224, 3),
                 n_class: int = 3,
                 batch_size: int = 32,
                 epochs: int = 10):
        self.input_shape = input_shape
        self.n_class = n_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.model_classification()
        self.weight_path = weight_path
        self.callbacks = self.get_callbacks()
        self.metrics = self.get_metrics()

    def model_classification(self) -> Model:
        """Modelo de classificação entre covid, normal e pneumonia

            Args:
                input_size (tuple, optional): Tamanho da imagem de entrada. Defaults to (224, 224, 3).
                n_class (int, optional): Número de classes de saída. Defaults to 3.

            Returns:
                (keras.Model) : Modelo do keras
        """
        resnet = ResNet50V2(include_top=False,
                            weights="imagenet",
                            input_shape=self.input_shape,
                            pooling="max")
        resnet.trainable = False
        output = Sequential([
            resnet,
            Dense(self.n_class, activation='softmax')])
        return output

    def get_callbacks(self):
        """
            Retorna a lista callbacks do modelo
            Returns:
                (list of keras.callbacks): lista dos callbacks
        """
        # Salva os pesos dos modelo para serem carregados
        # caso o monitor não diminua
        checkpoint = ModelCheckpoint(self.weight_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only=True)
        # Reduz o valor de LR caso o monitor nao diminuia
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.5,
                                      patience=3,
                                      verbose=1,
                                      mode='min',
                                      epsilon=1e-2,
                                      cooldown=2,
                                      min_lr=1e-8)
        # Parada do treino caso o monitor nao diminua
        early_stop = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   restore_best_weights=True,
                                   patience=40)
        # Termina se um peso for NaN (not a number)
        terminate = TerminateOnNaN()
        # Habilita a visualizacao no TersorBoard
        tensorboard = TensorBoard(log_dir="./logs")
        # Vetor a ser passado na função fit
        callbacks = [checkpoint,
                     early_stop,
                     reduce_lr,
                     terminate,
                     tensorboard]
        return callbacks

    def model_fit(self, dataset):
        """
            Realiza o treinamento do modelo
            Return:
                (list): metricas colocadas no compile
        """
        train, val, _ = dataset.step()
        train_x, train_y = train
        history = self.model.fit(x=train_x,
                                 y=train_y,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 callbacks=self.callbacks,
                                 validation_data=val)
        return history

    def model_compile(self):
        """Compila o modelo
        """
        self.model.compile(Adam(lr=2e-3),
                           loss='categorical_crossentropy',
                           metrics=self.metrics)
        self.model.summary()
        return None

    def get_metrics(self):
        """Gera as metricas para o modelo
            Returns:
                (list): metricas do modelo
        """
        metrics = ['accuracy']
        return metrics

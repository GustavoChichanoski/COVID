"""
    Biblioteca contendo as informações referente ao modelo.
"""
import cv2 as cv
import numpy as np
from keras.applications import ResNet50V2
from keras import Model
from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import TerminateOnNaN
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from src.model.grad_cam_split import grad_cam
from src.images.process_images import split_images_n_times as splits
from src.dataset.dataset import zeros
from src.images.read_image import read_images as ri
from src.images.process_images import normalize_image as ni
from src.images.process_images import resize_image as resize
from src.images.process_images import normalize_image as norma
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class ModelCovid:
    """[summary]
    """

    def __init__(self,
                 weight_path: str,
                 input_shape=(224, 224, 3),
                 batch_size: int = 32,
                 epochs: int = 10,
                 labels=['Covid', 'Normal', 'Pneumonia']):
        """
            Construtor da minha classe ModelCovid

            Args:
            -----
                weight_path (str): Caminho para salvar os pesos.
                input_shape (tuple, optional): Dimensão das imagens recortadas.
                                               Defaults to (224, 224, 3).
                batch_size (int, optional): pacotes por treinamento.
                                            Defaults to 32.
                epochs (int, optional): epocas de treinamento.
                                        Defaults to 10.
                labels (list, optional): Rotulos de saída.
                                         Defaults to ['Covid','Normal','Pneumonia'].
        """
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.labels = labels
        self.model = model_classification(self.input_shape,
                                          len(self.labels))
        self.weight_path = weight_path

    def model_save(self, history):
        """
            Salva os pesos do modelo em um arquivo
            Args:
                history (list): Historico das metricas de apreendizagem
        """
        val_acc = (1 - history['val_loss'][-1])*100
        self.model.save('model_acc_{:.02f}.h5'.format(val_acc),
                        overwrite=True)
        file_name_weight = 'weights_acc_{:.02f}.hdf5'.format(val_acc)
        self.model.save_weights(file_name_weight,
                                overwrite=False)
        print("Pesos salvos em {}".format(file_name_weight))

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
                                 callbacks=get_callbacks(self.weight_path),
                                 validation_data=val)
        return history

    def model_compile(self):
        """Compila o modelo
        """
        self.model.compile(Adam(lr=2e-3),
                           loss='categorical_crossentropy',
                           metrics=get_metrics())
        self.model.summary()

    def model_predict(self, image, n_splits: int = 100):
        """
            Realiza a predição do modelo

            Args:
                image ([type]): [description]
                n_splits (int, optional): [description]. Defaults to 100.

            Returns:
                [type]: [description]
        """
        image = ri(image)
        cuts, positions = splits(image, n_splits)
        cuts = np.array(cuts)
        cuts = cuts.reshape((n_splits,
                             self.input_shape[0],
                             self.input_shape[1],
                             self.input_shape[2]))
        votes = self.model.predict(cuts)
        elect = winner(self.labels, votes)
        heatmap = grad_cam(cuts[5, :, :, :],
                           self.model)
        pb_grad = np.zeros((1024,1024))
        for cut, pos in zip(cuts,positions):
            start = pos[0]
            end = pos[1]
            heatmap  = grad_cam(cut,
                                self.model)
            pb_grad[start:start + 224,
                    end:end + 224] += heatmap
        pb_grad = np.uint8(pb_grad)
        jet = cm.get_cmap("jet")
        jet_color = jet(np.arange(256))[:,:3]
        jet_heatmap = jet_color[heatmap]

        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((1024,1024))
        jet_heatmap = img_to_array(jet_heatmap)
        
        superimposed_image = jet_heatmap * 0.4 + image
        superimposed_image = array_to_img(superimposed_image)

        plt.imshow(superimposed_image)
        plt.show()

        return elect


def model_classification(input_shape,
                         n_class) -> Model:
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
    resnet = ResNet50V2(include_top=False,
                        weights="imagenet",
                        input_shape=input_shape,
                        pooling="max")
    resnet.trainable = False
    output = Sequential([resnet,
                         Dense(n_class,
                               activation='softmax',
                               name='classifier')])
    return output


def winner(labels, votes):
    """
        Retorna o label da doenca escolhido

        Args:
            labels (list): nomes das classes
            votes (list): predicao das imagens

        Returns:
            elect (str): label escolhido pelo modelo
    """
    n_class = len(labels)
    poll = zeros(n_class)
    for vote in votes:
        poll += vote
    elect = labels[np.argmax(poll)]
    return elect


def get_metrics():
    """
        Gera as metricas para o modelo
        Returns:
            (list): metricas do modelo
    """
    metrics = ['accuracy']
    return metrics


def get_callbacks(weight_path):
    """
        Retorna a lista callbacks do modelo
        Returns:
            (list of keras.callbacks): lista dos callbacks
    """
    # Salva os pesos dos modelo para serem carregados
    # caso o monitor não diminua
    checkpoint = ModelCheckpoint(weight_path,
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

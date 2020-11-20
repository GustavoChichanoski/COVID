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
from dataset import zeros
from read_image import read_images as ri
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.utils import get_file
from tensorflow import GradientTape
from process_images import split_images_n_times as splits

SPLIT_SIZE = 224
SPLIT_CHANNEL = 3
N_CLASS = 3


class ModelCovid:

    def __init__(self,
                 weight_path: str,
                 input_shape=(224, 224, 3),
                 n_class: int = 3,
                 batch_size: int = 32,
                 epochs: int = 10,
                 labels=['Covid', 'Normal', 'Pneumonia']):
        self.input_shape = input_shape
        self.n_class = n_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.model_classification()
        self.weight_path = weight_path
        self.callbacks = self.get_callbacks()
        self.metrics = self.get_metrics()
        self.labels = labels

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
            Dense(self.n_class, activation='softmax', name='classifier')])
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

    def model_save(self, history):
        val_acc = (1 - history['val_loss'][-1])*100
        self.model.save(
            'model_acc_{:.02f}.h5'.format(val_acc),
            overwrite=True)
        file_name_weight = 'weights_acc_{:.02f}.hdf5'.format(val_acc)
        self.model.save_weights(
            file_name_weight,
            overwrite=False)
        print("Pesos salvos em {}".format(file_name_weight))
        return 0

    def make_grand_cam_heatmap(self, image):
        # https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/grad_cam.ipynb#scrollTo=Ka1RKYSI76Q0
        # Primeiro, nós criamos um modelo os mapas das imagens de
        # entrada para a ativacação da ultima camada de convolução
        last_conv_layer = self.model.get_layer('resnet50v2')
        last_conv_layer_model = Model(self.model.inputs,
                                      last_conv_layer.output)

        # Segundo, nós criamos um modelo com os mapas de ativação da ultmima
        # camada de convolução para as predições da classe final
        classifier_input = Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layers_names:
            x = model.get_layer(lyer_name)(x)
        classifier_model = Model(classifier_input, x)

        # Terceiro, nós calculamos o gradiente das classe topo de predições
        # para as imagens de saida, respeitando as ativações da ultima camada
        # de convoluções
        with GradientTape as tape:
            last_conv_layer_output = last_conv_layer_model(img)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_chanel = preds[:, top_preds_index]
        grads = tape.gradiente(top_class_chanel)
        last_conv_layer_output = self.important_channel(last_conv_layer_output)
        heatmap = self.create_heatmap(last_conv_layer_output)
        return heatmap

    def calc_gradient(self,):
        with GradientTape as tape:
            last_conv_output = last_conv_layer_model(img)
            tape.watch(last_conv_output)
            preds = classifier_model(last_conv_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_chanel = preds[:, top_preds_index]
        grads = tape.gradiente(top_class_chanel)

    def important_channel(self, last_conv_output):
        pooled_grads = tf.reduce_mean(grads, (0, 1, 2))
        last_conv_output = last_conv_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_output[:, :, i] * pooled_grads[i]
        return last_conv_output

    def create_heatmap(self, last_conv_output):
        heatmap = np.mean(last_conv_output, axis=-1)
        # Para a visualização do heatmap, ele foi normalizado de 0 a 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

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

    def model_predict(self, image, n_splits):
        image = ri(image)
        cuts, pos = splits(image, n_splits)
        cuts = np.array(cuts)
        cuts = cuts.reshape((n_splits,
                             self.input_shape[0],
                             self.input_shape[1],
                             self.input_shape[2]))
        votes = self.model.predict(cuts)
        poll = zeros(self.n_class)
        for vote in votes:
            poll += vote
        max_votes = np.max(poll)
        elect = []
        id_candidate = 0
        for candidate in poll:
            if candidate == max_votes:
                elect = self.labels[id_candidate]
                break
            id_candidate += 1
        return elect

    def get_metrics(self):
        """Gera as metricas para o modelo
            Returns:
                (list): metricas do modelo
        """
        metrics = ['accuracy']
        return metrics

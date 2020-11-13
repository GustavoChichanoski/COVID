# %% Import libraries
import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import h5py
# %% Keras imports
from keras import backend as K
from keras.models import Input
from keras.models import Model
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.metrics import TruePositives
from keras.metrics import TrueNegatives
from keras.metrics import FalseNegatives
from keras.metrics import FalsePositives
from keras.metrics import BinaryAccuracy
from keras.metrics import AUC
from keras.metrics import Recall
from keras.metrics import Precision
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
h5py.run_tests()

# %% Preprocessing Image function


def image_grayscale(image):
    '''
        Convete uma imagem de BGR para GRAY
        Args:
            image - Imagem a ser convertida (np.array)
        return:
            Imagem em preto e branco (np.array)
    '''
    cv_gray = cv.COLOR_BGR2GRAY
    return cv.cvtColor(image, cv_gray)


def image_rescale(image, scale=255.0):
    '''
        Args:
            image - np.array
            scale - np.float32
        return:
            image - np.array
    '''
    return image/scale


def equalize_histogram(image):
    '''
        Equaliza o histograma da imagem
        Args:
            np.array
        return:
            np.array
    '''
    return cv.equalizeHist(image)


def invert_image(image):
    '''
        Função para inverter a imagem
        Args:
            Recebe uma imagem de np.array e inverte os valores
        return:
            Retorna a imagem como sendo um vetor np.array
    '''
    return cv.bitwise_not(image)


def read_image(path):
    '''
        Função para realizar a leitura de uma imagem atráves do 
        caminho do arquivo.
        Args:
            Recebe uma String contendo o caminho do arquivo
            da imagem.
        return :
            Retorna a imagem como sendo um vetor np.array 
    '''
    return cv.imread(path)


def image_resize(image, size=256):
    '''
        Redimensiona a imagem para o tamanho de size
    '''
    return cv.resize(image, (size, size))


def normalize_image(image, dim):
    '''
        Normaliza a imagem
        Args:
            image - np.array
        return:
            image - np.array
    '''
    image = image_resize(image, dim)
    image = image_grayscale(image)
    image = equalize_histogram(image)
    return image_rescale(image)


def Up_plus_Concatenate(layer, connection, i):
    # Define names of layers
    up_name = 'UpSampling{}_1'.format(i)
    conc_name = 'UpConcatenate{}_1'.format(i)
    # Create the layer sequencial
    layer = UpSampling2D(name=up_name)(layer)
    layer = Concatenate(axis=-1,
                        name=conc_name)([layer, connection])
    return layer


def conv_unet(
        layer,
        filters=32, kernel=(3, 3),
        act="relu",
        i=1, j=1):
    # Define os nomes das layers
    conv_name = "Conv{}_{}".format(i, j)
    bn_name = "BN{}_{}".format(i, j)
    act_name = "Act{}_{}".format(i, j)

    layer = Conv2D(
        filters=filters, kernel_size=kernel,
        padding='same', name=conv_name)(layer)
    # layer = BatchNormalization(name=bn_name)(layer)
    layer = Activation(act, name=act_name)(layer)
    return layer


def model_unet(
        input_size=(None, 256, 256, 1),
        depth=5,
        activation='relu',
        n_class=1,
        final_activation='sigmoid',
        filter_root=32):

    store_layers = {}

    inputs = Input(input_size)

    first_layer = inputs

    for i in range(depth):

        filters = (2**i) * filter_root

        # Cria as duas convoluções da camada
        for j in range(2):
            layer = conv_unet(
                first_layer, filters, (3, 3),
                activation, i, j)

        # Verifica se está na ultima camada
        if i < depth - 1:
            # Armazena a layer no dicionario
            store_layers[str(i)] = layer
            max_name = 'MaxPooling{}_1'.format(i)
            first_layer = MaxPooling2D(
                (2, 2), padding='same',
                name=max_name
            )(layer)

        else:
            first_layer = layer

    for i in range(depth-2, -1, -1):

        filters = (2**i) * filter_root
        connection = store_layers[str(i)]

        layer = Up_plus_Concatenate(first_layer, connection, i)

        for j in range(2, 4):
            layer = conv_unet(
                layer, filters, (3, 3),
                activation, i, j)

        first_layer = layer

    layer = Dropout(0.33, name='Drop_1')(layer)
    outputs = Conv2D(
        n_class, (1, 1), padding='same',
        activation=final_activation, name='output')(layer)

    return Model(inputs, outputs, name="UNet")


def dice_coef(y_true, y_pred):
    ''' Dice Coefficient
    Project: BraTs   Author: cv-lee   File: unet.py    License: MIT License
    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    Returns:
        (np.array): Calcula a porcentagem de acerto da rede neural
    '''

    class_num = 1

    for class_now in range(class_num):

        # Converte y_pred e y_true em vetores
        y_true_f = K.flatten(y_true[:, :, :, class_now])
        y_pred_f = K.flatten(y_pred[:, :, :, class_now])

        # Calcula o numero de vezes que
        # y_true(positve) é igual y_pred(positive) (tp)
        intersection = K.sum(y_true_f * y_pred_f)
        # Soma o número de vezes que ambos foram positivos
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        # Smooth - Evita que o denominador fique muito pequeno
        smooth = K.constant(1e-6)
        # Calculo o erro entre eles
        num = (K.constant(2)*intersection + 1)
        den = (union + smooth)
        loss = num / den

        if class_now == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss

    total_loss = total_loss / class_num

    return total_loss


def dice_coef_loss(y_true, y_pred):
    accuracy = 1 - dice_coef(y_true, y_pred)
    return accuracy


def numpy_to_keras(nparray, dim):
    keras = np.array(nparray)
    keras = keras.reshape(1, dim, dim, 1)
    return keras


# %% processing image
DIM = 256
DIM_NEW = 512
weight_path = './.model/weight.h5'
# Metrica de salvamento
checkpoint = ModelCheckpoint(
    weight_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=True)
# Metrica para a redução do valor de LR
reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    mode='min',
    epsilon=1e-2,
    cooldown=2,
    min_lr=1e-8)
# Metrica para a parada do treino
early = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=40)
callbacks_list = [checkpoint, early, reduceLROnPlat]

model = model_unet(
    (DIM, DIM, 1), filter_root=16,
    depth=4, activation='relu')

metrics = [
    TruePositives(name='tp'),  # Valores realmente positivos
    TrueNegatives(name='tn'),  # Valores realmente negativos
    FalsePositives(name='fp'),  # Valores erroneamente positivos
    FalseNegatives(name='fn'),  # Valores erroneamente negativos
    BinaryAccuracy(name='accuracy')]

weight_path = './.model/weight_val_acc_96.34.h5'
model = model_unet((DIM, DIM, 1), filter_root=32, depth=5, activation='relu')
model.compile(
    optimizer=Adam(lr=1e-3),
    loss=dice_coef_loss,
    metrics=metrics)
model.summary()
model.load_weights(weight_path)

new_data = './data'
old_data = os.listdir('./old_data')

def numpy_to_cv(image):
    return image.astype(np.uint8)

def segmentation_lung(lung,mask):
    mask = (mask > 0.8).astype(np.float32)
    return lung*mask

# %%
for _type in old_data:
    type_path = os.path.join('./old_data', _type)
    id = 0
    for _class in os.listdir(type_path):
        class_path = os.path.join(type_path, _class)
        for image_path in os.listdir(class_path):
            _image_path = os.path.join(class_path, image_path)
            image = read_image(_image_path)
            lung = normalize_image(image, DIM)
            lung = 2*lung - 1
            
            lung = numpy_to_keras(lung, DIM)
            mask = model.predict(lung)
            mask = mask[0]
            
            lung = read_image(_image_path)
            lung = image_resize(lung, DIM_NEW)
            lung = image_grayscale(lung)
            mask = image_resize(mask, DIM_NEW)
            seg_lung = segmentation_lung(lung,mask)

            # fig, (ax1,ax2,ax3) = plt.subplots(figsize=(13,3),ncols=3)
            # plt_lung = ax1.imshow(lung, cmap='gray')
            # ax1.set_title('lung')
            # fig.colorbar(plt_lung, ax=ax1)
            # ax1.axis('off')

            # ax2.set_title('mask')
            # plt_mask = ax2.imshow(mask, cmap='gray')
            # fig.colorbar(plt_mask, ax=ax2)
            # plt.axis('off')

            # ax3.set_title('Segmentation Lung')
            # plt_seg = ax3.imshow(seg_lung, cmap='gray')
            # fig.colorbar(plt_seg, ax=ax3)
            # ax3.axis('off')
            # plt.show()
            new_path = '{}/{}/{}/{:04d}.png'.format(
                new_data, _type, _class, id)
            seg_lung = numpy_to_cv(seg_lung)
            cv.imwrite(new_path, seg_lung)
            id += 1
# %%
cv.imwrite('./data/test/002.png',seg_lung)

# %%
new_path
# %%

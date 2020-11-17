# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from keras import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.applications import ResNet50V2
from keras.metrics import BinaryAccuracy
from keras.metrics import FalseNegatives
from keras.metrics import FalsePositives
from keras.metrics import TrueNegatives
from keras.metrics import TruePositives
from keras.layers.normalization import BatchNormalization
from keras.layers import LocallyConnected2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# %% [code] Funções
def find_start(image_with_black, size=1024):
    """Encontra o primeiro pixel não zero da esquerda para a direita.

    Args:
        image_with_black (np.array): Imagem a ser analizada.
        size (int, optional): Tamanho da imagem a ser analizada. Defaults to 1024.

    Returns:
        (tuple): Primeira linha e coluna contendo um pixel não zero.
    """
    rows, columns = [], []
    for image in image_with_black:
        for row in range(size):
            if np.sum(image_with_black[row]) > 0:
                break
        for column in range(size):
            if np.sum(image_with_black[:, column]) > 0:
                break
        rows.append(row)
        columns.append(column)
    return (rows,columns)

def find_end(image_with_black, size=1024):
    """ Encontra o primeiro pixel não zero da direita para a esquerda

    Args:
        image_with_black (np.array): imagem a ser analizada
        size (int, optional): tamanho da imagem a ser analizada. Defaults to 1024.

    Returns:
        (tuple): Primeira linha e coluna contendo um pixel não zero.
    """
    rows, columns = [], []
    for image in image_with_black:
        for row in range(size-1, 0, -1):
            if np.sum(image[row]) > 0:
                break
        for column in range(size-1, 0, -1):
            if np.sum(image[:, columnj]) > 0:
                break
        rows.append(row)
        columns.append(columns)
    return rows, columns

def gray_to_RGB(gray_image):
    """Transforma imagens em escala de cinza em coloridas.

    Args:
        gray_image (np.array): Imagem em escala de cinza.

    Returns:
        (np.array): Imagens colorida.
    """
    coloreds = []
    for image in gray_image:
        colored = cv.cvtColor(gray_image, cv.COLOR_GRAY2RGB)
        coloreds.append(colored)
    return coloreds

def split_images_without_black(
        image,
        start=(0, 0),
        end=(1024, 1024),
        n_split=100,
        dim_split=224):
    """ Recorta a imagem em n_split vezes de tamanhos dim_split começando
        start até end ignorando recortes totalmente pretos.

    Args:
        image (np.array): imagem a ser recortada
        start (tuple, optional): comeco da imagem onde aparece algo. Defaults to (0, 0).
        end (tuple, optional): fim da imagem onde aparece algo. Defaults to (1024, 1024).
        n_split (int, optional): numero de vezes a serem cortadas. Defaults to 100.
        dim_split (int, optional): tamanho dos recortes. Defaults to 224.

    Returns:
        [np.array]: recortes das imagens
    """    
    split_images, x_split, y_split = [], [], []

    for _ in tqdm(range(n_split)):

        x, y = random_xy(start, end, dim_split)

        recort = image[x:x+dim_split, y:y+dim_split]
        while(np.sum(recort) < 255):
            x, y = random_xy(start, end, dim_split)
            recort = image[x:x+dim_split, y:y+dim_split]
        split_images.append(recort)
        x_split.append(x)
        y_split.append(y)

    return split_images, x_split, y_split

def normalize_image(recorts):
    """Normaliza os recortes para que todos variem de -1 a 1.

    Args:
        recorts (np.array): Imagens recortadas originais, pode ser uma lista de imagens.

    Returns:
        (np.array): Imagens recortadas normalizadas
    """
    normalize = []
    for recort in recorts:
        recort = recort/np.max(recort)
        recort = 2*recort - 1
        normalize.append(recort)
    return normalize

def random_xy(start=(0,0), end=(0,0), dim_split=224):
    """ Criam x e y randomicamente comecando de start até end menos a dimensão maxima do corte

    Args:
        start (tuple, optional): Valores iniciais de x e y. Defaults to (0,0).
        end (tuple, optional): Valores finais de x e y. Defaults to (0,0).
        dim_split (int, optional): Dimensão do recorte. Defaults to 224.

    Returns:
        (tuple): x e y gerados aleatoriamente
    """    
    x_i, y_i = start
    x_e, y_e = end
    x = np.random.randint(x_i, x_e-dim_split)
    y = np.random.randint(y_i, y_e-dim_split)
    return x, y

def rescale_image(original_image, scale=255):
    """ Rescala a imagem para ir de -1 a 1

    Args:
        original_image (np.array): imagem ainda não rescalada
        scale (int, optional): escala da nova imagem. Defaults to 255.

    Returns:
        (np.array) : imagem rescalada
    """
    half_scale = scale/2
    return (original_image-half_scale)/half_scale

def reads_images_by_id(images_paths='./data/img.png', id_start=0, id_end=-1):
    """ Lê as imagens do listas de caminhos da imagem de start até end -1

    Args:
        images_paths (str): Arrays contndo os caminhos das imagens. Defaults to './data/img.png'.
        id_start (int,optional): ID de inicio das leituras das imagens. Defaults to 0.
        id_end (int, optional): ID de termino das leituras das imagens, caso não seja passado todas as imagens a partir de id_start serão lidas. Defaults to -1.

    Returns:
        imagens: retorna uma lista np.array das imagens lidas
    """
    reads_images = []
    if(end < start):
        end = len(images_paths)

    for i in range(id_start, id_end):
        img = cv.imread(images_paths[i])
        img = rescale_image(img, 127)
        reads_images.append(img)

    return reads_images

def model_classification(input_size=(224, 224, 3), n_class=3):
    """Modelo de classificação entre covid, normal e pneumonia

    Args:
        input_size (tuple, optional): Tamanho da imagem de ebtrada. Defaults to (224, 224, 3).
        n_class (int, optional): Número de classes de saída. Defaults to 3.

    Returns:
        (keras.Model) : Modelo do keras
    """
    resnet = ResNet50V2(include_top=False,
                        weights="imagenet",
                        pooling='avg')
    resnet.trainable = False
    output = Sequential([resnet, Dense(n_class, activation='softmax')])
    return output

# %% [code]
DIM = 1024
DIM_SPLIT = 224
K_SPLIT = 100
# %% [code]
DATA = "../input/lung-segmentation-1024x1024/data"
path_test = os.path.join(DATA, 'test')
path_train = os.path.join(DATA, 'train')
class_names = os.listdir(path_train)
# %% [code]
url_lung = '../input/lung-segmentation-1024x1024/data/test/Pneumonia/0833.png'
image = cv.imread(url_lung)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
# %% [code]
x_start, y_start = find_start(image, DIM)
x_end, y_end = find_end(image, DIM)
image_without_black = image[x_start:x_end, y_start:y_end]
plt.imshow(image_without_black, cmap='gray')
plt.show()

start = (x_start, y_start)
end = (x_end, y_end)

# %% [code]
recort_images, x_split, y_split = split_images_without_black(
    image, start, end, K_SPLIT, DIM_SPLIT)
norm_recort = np.array(normalize_image(recort_images))

# %% [code]
print(norm_recort.shape)

# %% [code]
# for recort in norm_recort:
#     plt.imshow(recort,cmap='gray')
#     plt.show()

# %% [code]
zeros = np.zeros((DIM, DIM))
ones = np.ones((DIM_SPLIT, DIM_SPLIT))

for i in range(K_SPLIT):
    zeros[x_split[i]:x_split[i]+DIM_SPLIT,
          y_split[i]:y_split[i] + DIM_SPLIT] += ones

passou = np.where(zeros != 0, 1, 0)
passou_image = image*passou
plt.imshow(passou, cmap='gray')
plt.show()
plt.imshow(image, cmap='gray')
plt.show()
plt.imshow(passou_image, cmap='gray')
plt.show()
# %% [code]
recort_images, x_split, y_split = split_images_without_black(
    image, start, end, K_SPLIT, DIM_SPLIT)

# %% [code]
DATA_TRAIN = os.path.join(DATA, 'train')

# %% [code]
classes = os.listdir(DATA_TRAIN)
n_classes = len(classes)

# %% [code]
model = model_classification(n_class=3)
model.compile(
    optimizer=Adam(lr=1e-3),
    loss='binary_crossentropy',
    metrics=[BinaryAccuracy(name='accuracy')]
)
model.summary()

# %% [code]
covid_train_p = os.path.join(DATA_TRAIN, 'Covid')
normal_train_p = os.path.join(DATA_TRAIN, 'Normal')
pneum_train_p = os.path.join(DATA_TRAIN, 'Pneumonia')

# %% [code]
covid_url = os.listdir(covid_train_p)
normal_url = os.listdir(normal_train_p)
pneumonia_url = os.listdir(pneum_train_p)

# %% [code]
total = len(covid_url) + len(normal_url) + len(pneumonia_url)

# %% [code]
rel = [len(covid_images)/total, len(normal_images) /
       total, len(pneumonia_images)/total]
# %% [code]
n_step = 10
covid_id = 0
normal_id = 0
pneumonia_id = 0

while(covid_id < len(covid_images)):

    covid_id += np.floor(n_step*rel[0]).astype(np.int32)
    normal_id += np.floor(n_step*rel[1]).astype(np.int32)
    pneumonia_id += np.floor(n_step*rel[2]).astype(np.int32)

    print("covid: {:4d}/{} normal: {:4d}/{} pneumonia: {:4d}/{}".format(
        covid_id, len(covid_images),
        normal_id, len(normal_images),
        pneumonia_id, len(pneumonia_images)))

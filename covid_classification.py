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
    if isinstance(image_with_black,list):
        start = []
        for img in image_with_black:
            row, column = find_start(img,size)
            start.append((row,column))
        return start
    else:
        row_start, column_start = 0,0
        for i in range(size):
            if np.sum(image_with_black[i]) > 0:
                    row_start = i
                    break
        for j in range(size):
            if np.sum(image_with_black[:, j]) > 0:
                column_start = j
                break
        return row_start, column_start

def find_end(image_with_black, size=1024):
    """ Encontra o primeiro pixel não zero da direita para a esquerda

    Args:
        image_with_black (np.array): imagem a ser analizada
        size (int, optional): tamanho da imagem a ser analizada. Defaults to 1024.

    Returns:
        (tuple): Primeira linha e coluna contendo um pixel não zero.
    """
    if isinstance(image_with_black,list):
        ends = []
        for image in image_with_black:
            row, column = find_end(image,size)
            ends.append((row,column))
        return ends
    else:
        row_end, column_end = 0, 0
        for i in range(size - 1, -1, -1):
            if np.sum(image_with_black[i]) > 0:
                row_end = i
                break
        for j in range(size - 1, -1, -1):
            if np.sum(image_with_black[:, j]) > 0:
                column_end = j
                break
        return row_end, column_end

def gray2rgb(gray_image):
    """Transforma imagens em escala de cinza em coloridas.

    Args:
        gray_image (np.array): Imagem em escala de cinza.

    Returns:
        (np.array): Imagens colorida.
    """
    if isinstance(gray_image,list):
        coloreds = []
        for gray in gray_image:
            colored = gray2rgb(gray)
            coloreds.append(colored)
        return coloreds
    else:
        return cv.cvtColor(gray, cv.COLOR_GRAY2RGB)

def bgr2gray(colored_images):
    """Transforma imagens coloridas em escala de cinza.

    Args:
        colored_images (np.array): Imagem colorida.

    Returns:
        (np.array): Imagens em escala cinza.
    """
    if isinstance(colored_images,list):
        grays = []
        for color in colored_images:
            gray = bgr2gray(color)
            grays.append(gray)
        return grays
    else:
        return cv.cvtColor(colored_images, cv.COLOR_BGR2GRAY)

def split_images_n_times(image,n_split=100,dim_orig=1024,dim_split=224):
    """ Recorta a imagem em n_split vezes de tamanhos dim_split começando start até end ignorando recortes totalmente pretos.

    Args:
        image (np.array): imagem a ser recortada
        n_split (int, optional): numero de vezes a serem cortadas. Defaults to 100.
        dim_orig (int, optional): tamanho original  da imagem. Defaults to 1024.
        dim_split (int, optional): tamanho dos recortes. Defaults to 224.

    Returns:
        (tuple): recortes das imagens, x inicias dos recortes e y iniciais dos recortes.
    """    
    if not isinstance(image,list):
        split_images, split_x, split_y = [], [], []
        lung_start = find_start(image,dim_orig)
        lung_end = find_end(image,dim_orig)
        print(lung_start)
        print(lung_end)
        for _ in tqdm(range(n_split)):

            pos = random_xy(lung_start, lung_end, dim_split)
            recort = create_recort(image,pos,dim_split)

            while(np.sum(recort) < 255):
                pos = random_xy(lung_start, lung_end, dim_split)
                recort = create_recort(image,pos,dim_split)

            split_images.append(recort)
            split_x.append(pos[0])
            split_y.append(pos[1])

        return split_images, split_x, split_y
    else:
        split_image, split_x, split_y = [], [], []
        for img in image:
            img_split_images, img_split_x, img_split_y = split_images_n_times(
                img, n_split, dim_orig, dim_split)
            split_image.append(img_split_images)
            split_x.append(img_split_x)
            split_y.append(img_split_y)
        return split_image, split_x, split_y

def create_recort(image,pos_start=(0,0),dim_split=224):
    """Cria um recorte da imagem indo da posicao inicial até a dimensão do recorte

    Args:
        image (np.array): [description]
        pos_start (tuple, optional): Posicao do recorte. Defaults to (0,0).
        dim_split (int, optional): Dimensão do recorte. Defaults to 224.
    Return:
        (np.array): Recorte da imagem
    """
    pos_end = (pos_start[0]+dim_split, pos_start[1]+dim_split)
    return image[pos_start[0]:pos_end[0],pos_start[1]:pos_end[1]]

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
    """ Criam x e y randomicamente comecando de start até end menos a dimensão maxima do corte.

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

def rescale_images(original_image, scale=255):
    """ Rescala a imagem para ir de -1 a 1

    Args:
        original_image (list or np.array): imagem ainda não rescalada
        scale (int, optional): escala da nova imagem. Defaults to 255.

    Returns:
        (list or np.array) : imagem rescalada
    """
    if isinstance(original_image, list):
        rescales = []
        for img in original_image:
            rescales.append(rescale_image(img,scale))
        return rescales
    return rescale_image(original_image, scale)

def read_random_image(paths: list,id_start:list = [0,1]) -> list:
    """ Lê as imagens dos ids contidos em id_start

    Args:
        paths (list): Caminhos completos das imagens a serem lidas.
        id_start (list, optional): ids das imagens a serem lidas. Defaults to [0,1].

    Returns:
        list: lista das imagens lidas.
    """
    images = []
    for i in id_start:
        images.append(read_images(paths[i]))
    return images

def read_sequencial_image(paths: list,id_start:int = 0,id_end:int = 1) -> list:
    """Lê sequencialmente as imagens

    Args:
        paths (list): Caminhos completos das imagens a serem lidas.
        id_start (int, optional): ID inicial da imagem a ser lida. Defaults to 0.
        id_end (int, optional): ID final da imagem a ser lida. Defaults to 0.

    Returns:
        list: lista das imagens lidas
    """
    images = []
    for i in range(id_start, id_end):
        images.append(read_images(paths[i]))
    return images

def read_images(images_paths='./data/img.png', id_start=0, id_end=-1):
    """ Lê as imagens do listas de caminhos da imagem de start até end -1

    Args:
        images_paths (str or list): Arrays contndo os caminhos das imagens. Defaults to './data/img.png'.
        id_start (int or list,optional): ID do inicio das imagens. Defaults to 0.
        id_end (int, optional): ID do fim das imagens, caso não seja passado todas as imagens a depois de id_start serão lidas. Defaults to -1.

    Returns:
        (np.array or list): retorna uma lista np.array das imagens lidas
    """
    if isinstance(images_paths,list) :
        if isinstance(id_start,int):
            if id_end < id_start :
                id_end = len(images_paths)
            return read_sequencial_image(images_paths,id_start,id_end)
        else:
            return read_random_image(images_paths,id_start)
    else:
        return cv.imread(images_paths)

def plot_images(images,cmap=None):
    """Plotas as imagens passafas em images

    Args:
        images (list or np.array): imagens a serem plotadas
    """    
    if isinstance(images,list):
        for img in images:
            plot_images(img,cmap)
    else:
        if cmap == 'gray':
            plt.imshow(images,'gray')
        else:
            plt.imshow(images)
        plt.show()

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

def listdir_full_path(path='./data/Covid/0000.png'):
    """ É um os.listdir só que retornando todo o caminho do arquivo.

    Args:
        path (str): caminho pai dos arquivos

    Returns:
        (list): lista de strings contendo o caminho todo das imagens.
    """
    urls = os.listdir(path)
    full_path = [os.path.join(path,url) for url in urls]
    return full_path

# %% [code] Definindo as constantes do projeto
DIM_ORIGINAL = 1024
DIM_SPLIT = 224
K_SPLIT = 100
KAGGLE = False
if KAGGLE :
    DATA = "../input/lung-segmentation-1024x1024/data"
else:
    DATA = "./data"
TRAIN_PATH = os.path.join(DATA,'train')
TEST_PATH = os.path.join(DATA, 'test')
# %% [markfdown] Verificando o funcionamento das funções
# %% [code] Verificando o funcionamento da funcao read_images
covid_example = os.path.join(TRAIN_PATH, "Covid/0000.png")
# Testando para uma imagem
print("Lendo apenas uma imagem")
image_covid_example = read_images(covid_example)
plt.imshow(image_covid_example)
plt.show()
# Testando para todas as imagens
covid_path = os.path.join(TRAIN_PATH, "Covid")
covid_urls = listdir_full_path(covid_path)
len_covid_urls = len(covid_urls)
print("Lendo de {} até {}".format(len_covid_urls - 2, len_covid_urls))
covid_images = read_images(covid_urls, len_covid_urls - 2)
plot_images(covid_images)
print("Lendo de 0 até 1")
covid_images = read_images(covid_urls, 0,2)
plot_images(covid_images)
print("Lendo as imagens 10 e 0")
covid_images = read_images(covid_urls, [10,0])
plot_images(covid_images)
# read_images funcionando
# %% Verificando o funcionamento da funçao bgr2gray
print("Convertendo para cinza")
gray_image = bgr2gray(covid_images)
plot_images(gray_image, 'gray')
gray_image = bgr2gray(image_covid_example)
plot_images(gray_image, 'gray')
print("Colorido para cinza funcionando")
# %% Verificando o funcionamento da funçao rescale_image
rescale_example = rescale_images(covid_images)
print(np.max(rescale_example[1]),np.min(rescale_example[1]))
print("Rescalonamento funcionando")
# %% Verificando a função split_images_without_black
print("Recortando a imagem")
recorts, x, y = split_images_n_times(gray_image)
plot_images(recorts,'gray')
print("Recorte funcionando")
# %%

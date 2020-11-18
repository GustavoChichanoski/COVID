# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import cv2 as cv
import matplotlib.pyplot as plt
from process_images import split_images_n_times
from read_image import read_images as ri
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

def proportion_class(path_type:str) -> list:
    """[summary]

    Args:
        path_type (str): O caminho onde as imagens estão separadas em classes.

    Returns:
        list: Retorna uma lista contendo a proporção de imagens no dataset.
    """
    diseases = [len(os.listdir(disease)) for disease in listdir_full_path(path_type)]
    total = 0
    for disease in diseases:
        total += disease
    proportion = np.array(diseases)/total
    return proportion

def number_images_load_per_step(path_type: str, img_loads=10) -> list:
    """Retorna a proporção de imagens que devem ser carregadas por classes a cada passo.

    Args:
        path_type (str): caminho contendo as classes
        img_loads (int, optional): Número de imagens a ser carregadas por passo. Defaults to 10.

    Returns:
        list: numero de imagens que devem ser carregadas por classe.
    """
    proportion = proportion_class(path_type)
    img_ready_load = img_loads
    img_per_class = []
    for p_class in proportion:
        img_per_class.append(np.floor(p_class*img_loads))
        img_ready_load -= img_per_class[-1]
    img_per_class[-1] += img_ready_load
    return img_per_class
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

COVID_PATH = os.path.join(TRAIN_PATH,'Covid')
NORMAL_PATH = os.path.join(TRAIN_PATH, 'Normal')
PNEUM_PATH = os.path.join(TRAIN_PATH, 'Pneumonia')
# %% Calculo da proporcao
path_image = os.path.join(COVID_PATH,"0000.png")
image = ri(path_image)
cuts,pos = split_images_n_times(image)
plot_images(cuts)
# %%
# %% Lendo as imagens segundo a proporção

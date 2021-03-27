# %% [code]
"""
    O Objetivo desse programa é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
# %% [code] Constante
import sys
from src.zip.zip_save import zipfolder
from src.plots.graph import plot_dataset
from src.model.model import ModelCovid
from src.dataset.dataset import Dataset
from src.model.generator import DataGenerator
from os.path import join
from os import listdir
import numpy as np
import os
# %% Constante
DIM_ORIGINAL = 1024
DIM_SPLIT = 224
K_SPLIT = 100
# %% Imports
# -*- coding: utf-8 -*-
__version__ = '1.0'
KAGGLE = False
if os.path.exists('../input'):
    __SYS = join('../input', listdir('../input')[0])
    sys.path.append(__SYS)
    KAGGLE = True
else:
    __SYS = './'
# %% [code] Paths
DATA = join(__SYS, 'data')
TEST = join(DATA, 'test/Covid/0000.png')
WEIGHT = join(__SYS, 'weights')
NETS = os.listdir(WEIGHT)
TRAIN_PATH = join(DATA, 'train')
TEST_PATH = join(DATA, 'test')
COVID_PATH = join(TRAIN_PATH, 'Covid')
NORMAL_PATH = join(TRAIN_PATH, 'Normal')
PNEUM_PATH = join(TRAIN_PATH, 'Pneumonia')
# %% [code] Criação do dataset
nets_p = [join(WEIGHT, net) for net in NETS]
pesos = []
for rede in NETS:
    parent_net = join(WEIGHT, rede)
    net = []
    for pesos_path in listdir(parent_net):
        net.append(join(parent_net, pesos_path))
    pesos.append(net)
np.random.seed(seed=42)
labels = listdir(TRAIN_PATH)
params = {'labels': labels, 'dim': 224, 'batch_size': 32,
          'train': True, 'n_class': len(labels),
          'shuffle': True, 'channels': 3}
dataset = Dataset(path_data=DATA)
train, val = dataset.partition()
train_generator = DataGenerator(data=train, **params)
val_generator = DataGenerator(data=val, **params)
for net in range(len(NETS)):
    model = NETS[net]
    path = pesos[net][-1]
    # %% [code] Criação do modelo
    covid = ModelCovid('.model/weights.best.hfd5',
                       labels=labels,
                       epochs=100,
                       model=model,
                       batch_size=16)
    # %% Compilação do modelo
    covid.compile(loss='categorical_crossentropy', lr=1e-5)
    # covid.model.summary()
    # covid.model.layers[0].summary()
    # %% Apreendendo
    covid.load(path)
    history = []
    history = covid.fit_generator(train_generator=train_generator,
                                  val_generator=val_generator)
    path = covid.save(path, model=model, history=history)
    # %% [code] Carregamento do modelo
    covid.load(path)
    # %% [code] Predição da imagem
    covid.predict(TEST, 1)
    test = Dataset(path_data=DATA, train=False)
    test_v, val_v = test.partition(0.000001)
    test2 = DataGenerator(test_v, **params)
    matrix = covid.confusion_matrix(test2.x, 1)
    # %%
    fig_path = '{}'.format(join(nets_p[net], 'fig'))
    plot_dataset(names=labels, absolut=matrix,
                 n_images=1, path=fig_path)

zipfolder('./', 'fig.zip')

# %% [code]
"""
    O Objetivo desse programa é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
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

DIM_ORIGINAL = 1024
DIM_SPLIT = 224
K_SPLIT = 100
BATCH_SIZE = 32

__version__ = '1.0'

KAGGLE = False
if os.path.exists('../input'):
    __SYS = join('../input', listdir('../input')[0])
    sys.path.append(__SYS)
    KAGGLE = True
else:
    __SYS = './'

DATA = join(__SYS, 'data')
TRAIN_PATH = join(DATA, 'train')
TEST_PATH = join(DATA, 'test')
FIG_PATH = './fig'
WEIGHT = join(__SYS, 'weights')
NETS = os.listdir(WEIGHT)

TEST = join(DATA, 'test/Covid/0000.png')
# %% [code] Criação do dataset
np.random.seed(seed=42)

nets_path = [join(WEIGHT, net) for net in NETS]
pesos = []
for rede in NETS:
    parent_net = join(WEIGHT, rede)
    net = []
    for pesos_path in listdir(parent_net):
        net.append(join(parent_net, pesos_path))
    pesos.append(net)

labels = listdir(TRAIN_PATH)

dataset = Dataset(path_data=DATA)
train, val = dataset.partition()

params = {'labels': labels, 'dim': DIM_SPLIT,
          'batch_size': BATCH_SIZE,
          'train': True, 'n_class': len(labels),
          'shuffle': True, 'channels': 3}
train_generator = DataGenerator(data=train, **params)
val_generator = DataGenerator(data=val, **params)

for path_list, model, net_path in zip(pesos, NETS, nets_path):
    path = path_list[-1]

    covid = ModelCovid('.model/weights.best.hfd5',
                       labels=labels, epochs=100, model=model,
                       batch_size=BATCH_SIZE)

    covid.compile(loss='categorical_crossentropy', lr=1e-5)

    # covid.load(path)

    # history = covid.fit_generator(train_generator=train_generator,
    #                               val_generator=val_generator)
    path = covid.save(path, model=model, history=history)

    covid.load(path)
    n = 100
    name = join(FIG_PATH,
                join(model,'{}_{}'.format(model,n)))
    covid.predict(image=TEST,n_splits=n,name=name)

    test = Dataset(path_data=DATA, train=False)
    test_values, _val_v = test.partition(1e-5)
    test_generator = DataGenerator(test_values, **params)

    # matrix = covid.confusion_matrix(test_generator.x, 1)

    # fig_path = join(net_path, 'fig')
    # plot_dataset(names=labels, absolut=matrix,
    #              n_images=1, path=fig_path)

zipfolder('./', 'fig.zip')

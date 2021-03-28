# %% [code]
"""
    O Objetivo desse programa é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
from src.output_result.folders import create_folders
import sys
from src.output_result.zip_save import zipfolder
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

NETS = ['DenseNet201',
        'InceptionResNetV2',
        'ResNet50V2',
        'VGG19']

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

OUTPUT_PATH = 'outputs'

paths = create_folders(name=OUTPUT_PATH,
                       parent='./',
                       nets=NETS)

nets_path, net_weights, net_figures = paths

TEST = join(DATA, 'test/Covid/0000.png')
# %% [code] Criação do dataset
np.random.seed(seed=42)

trained = True

pesos = []
for net_weight in net_weights[0:1]:
    net = []
    list_train = listdir(net_weight)
    if len(list_train) > 0:
        for pesos_path in list_train:
            net.append(join(net_weight, pesos_path))
    else:
        trained = False
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

for path_list, model, net_path, net_figure in zip(pesos,
                                      NETS,
                                      nets_path,
                                      net_figures):
    path = path_list[-1]

    covid = ModelCovid('.model/weightss.best.hfd5',
                       labels=labels, epochs=100, model=model,
                       batch_size=BATCH_SIZE)

    covid.compile(loss='categorical_crossentropy', lr=1e-5)

    if trained:
        covid.load(path)
    else:
        history = covid.fit_generator(train_generator=train_generator,
                                      val_generator=val_generator)
        path = covid.save(path, model=model, history=history)

    covid.load(path)
    n = 1
    name = join(net_figure,
                '{}_{}'.format(model, n))
    # covid.predict(image=TEST, n_splits=n, name=name,grad=False)

    # test = Dataset(path_data=DATA, train=False)
    # test_values, _val_v = test.partition(1e-5)
    # test_generator = DataGenerator(test_values, **params)

    # matrix = covid.confusion_matrix(test_generator.x, 1)
    matrix = np.array([[[3,3,3],[3,3,3],[3,3,3]]])
    plot_dataset(names=labels, absolut=matrix,
                 n_images=1, path=net_figure)

zipfolder('./outputs', 'output.zip')

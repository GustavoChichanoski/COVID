"""
    O Objetivo desse código é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
# -*- coding: utf-8 -*-
import os
from src.dataset import dataset as ds
from src.model.model import ModelCovid

__version__ = '0.1'

DIM_ORIGINAL = 1024
DIM_SPLIT = 224
K_SPLIT = 100
KAGGLE = False

if KAGGLE:
    DATA = "../input/lung-segmentation-1024x1024/data"
else:
    DATA = "./data"

TRAIN_PATH = os.path.join(DATA, 'train')
TEST_PATH = os.path.join(DATA, 'test')

COVID_PATH = os.path.join(TRAIN_PATH, 'Covid')
NORMAL_PATH = os.path.join(TRAIN_PATH, 'Normal')
PNEUM_PATH = os.path.join(TRAIN_PATH, 'Pneumonia')

# %%
dataset = ds.Dataset(DATA)

covid = ModelCovid('.model/weights.best.hfd5',
                   labels=os.listdir(TRAIN_PATH),
                   epochs=1)
covid.model_compile()
# %% Apreendendo
covid.model_fit(dataset)
# %%
print(covid.model_predict('./data/test/Covid/0281.png',
                          500))

# %%

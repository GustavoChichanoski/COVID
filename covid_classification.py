"""
    O Objetivo desse código é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
try:
    import sys
    sys.path.insert(0, './src')
except:
    pass
import dataset as ds
from model import ModelCovid
import os
import numpy as np

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
                   n_class=len(dataset.folder_names))
covid.model_compile()
covid.model_fit(dataset)

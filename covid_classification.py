"""
    O Objetivo desse código é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
# %% [code]
import os
import sys
sys.path.insert(0,'./src')

import dataset as ds
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

# %%
dataset = ds.Dataset(DATA)
features = dataset.get_features_per_steps()
print(features.shape)
# %%

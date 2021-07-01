# %%
from typing import List, Union
from src.models.segmentacao.segmentacao_model import Unet
from src.output_result.folders import get_all_files_in_folder
from src.dataset.generator_seg import SegmentationDatasetGenerator
from pathlib import Path
from tensorflow.python import keras as keras
import matplotlib.pyplot as plt

DIM = 512
BATCH_SIZE = 1

model = Unet(dim=DIM,final_activation='sigmoid')
model.compile(loss='log_cosh_dice',rf=10)
model.build()
model.summary()

old_data = Path('D:\\Mestrado\\old_data')
new_data = Path('D:\\Mestrado\\new_data')
paths = get_all_files_in_folder(old_data)
peso = 'D:\Mestrado\pesos\pesos.hdf5'

model.load_weights(peso)

paths =  paths[:10]
params = {'batch_size': BATCH_SIZE, 'dim': 512}
datas = SegmentationDatasetGenerator(paths,None, **params)

predicts = model.predict(datas)
for predict in predicts:
    plt.imshow(predict,cmap='gray')
    plt.show()
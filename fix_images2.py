# %%
import matplotlib.pyplot as plt
# %%
from typing import List, Union
from src.models.segmentacao.segmentacao_model import Unet
from src.output_result.folders import get_all_files_in_folder
from src.dataset.generator_seg import SegmentationDatasetGenerator
from pathlib import Path
from tensorflow.python import keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import numpy as np

DIM = 256
ORI = 1024
BATCH_SIZE = 1

model = Unet(dim=DIM,final_activation='sigmoid')
model.compile(loss='log_cosh_dice',rf=1)
model.build()
model.summary()

old_data = Path('D:\\Mestrado\\old_data')
new_data = Path('D:\\Mestrado\\new_data')
paths = get_all_files_in_folder(old_data)
peso = 'D:\Mestrado\pesos\pesos2.hdf5'

model.load_weights(peso)

# paths = paths[:10]
params = {'batch_size': BATCH_SIZE, 'dim': DIM}
datas = SegmentationDatasetGenerator(paths,None, **params)

# %%
i = 0
for data, path in zip(datas,paths):
    predict = model.predict(data)
    seg_lung_image_path = str(new_data / '/'.join(path.parts[-3:-1]) / f'{i:04}.png')
    i += 1
    img = (predict[0] > 0.4).astype(np.uint8)
    img = cv.resize(img,(ORI,ORI))
    ori = data[0]
    ori = cv.resize(ori,(ORI,ORI))
    img = cv.bitwise_and(ori,ori,mask=img)
    img *= 255
    cv.imwrite(seg_lung_image_path,img)
# %%

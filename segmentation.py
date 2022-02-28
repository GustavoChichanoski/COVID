# %%
from pathlib import Path

from numpy import random
from src.models.segmentation.unet_functional import (
  unet_compile,
  unet_fit,
  unet_functional,
)
from src.dataset.segmentation.dataset_seg import SegmentationDataset
from src.dataset.segmentation.generator_seg import SegmentationDatasetGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = Path("C:\\Users\\Gustavo\\Projetos\\COVID\\dataset")
lung = data / "lungs"
mask = data / "masks"

BATCH_SIZE = 10
DIM = 256

dataset = SegmentationDataset(lung, mask)
train, val = dataset.partition(shuffle=False)

params = {"dim": DIM}

train_gen = SegmentationDatasetGenerator(
  train[0], train[1], augmentation=False, **params
)
val_gen = SegmentationDatasetGenerator(val[0], val[1], augmentation=False, **params)
model = unet_functional(
  inputs=(DIM, DIM, 1),
  filter_root=32,
  depth=5,
  activation="relu",
  final_activation="sigmoid",
  n_class=1,
  rate=0.3,
)

unet_compile(model=model, loss="log_cosh_dice", lr=1e-5, rf=1)
# history = unet_fit(model, train_gen, val_gen)

model.load_weights("C:\\Users\\Gustavo\\Projetos\\COVID\\pesos\\unet.h5")

test_params = {
  'batch_size': BATCH_SIZE,
  'dim': DIM,
  'flip_vertical': False,
  'flip_horizontal': False,
  'angle': 10.0,
  'load_image_in_ram': False,
  'threshold': 0.1,
}
test_generator = SegmentationDatasetGenerator(
  val[0], val[1], augmentation=False, **test_params
)
predicts = model.predict(test_generator)

# %%
i = 50
plt.figure(dpi=300)
plt.imshow(test_generator[i][0][:][:][0].reshape(DIM, DIM), cmap='gray')
plt.axis("off")
plt.show()

plt.figure(dpi=300)
plt.imshow(test_generator[i][1][:][:][0].reshape(DIM, DIM), cmap='gray')
plt.axis("off")
plt.show()

plt.figure(dpi=300)
plt.imshow(predicts[i][:][:].reshape(DIM, DIM) > 0.7, cmap='gray')
plt.axis("off")
plt.show()
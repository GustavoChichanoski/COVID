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

data = Path("D:\Mestrado\datasets\Lung Segmentation")
lung = data / "CXR_png"
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

model.load_weights("outputs\Segmentation\weights.h5")

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


# # %%
# history = pd.read_csv("outputs\Segmentation\working.csv")
# max_train = np.max(history["f1"].values) * 100
# max_val = np.max(history["val_f1"].values) * 100
# plt.figure(dpi=300)
# plt.title(f"Valor F1 do modelo ao longo das épocas.")
# plt.xlabel("Épocas")
# plt.ylabel("F1 [%]")
# plt.plot(history["f1"] * 100)
# plt.plot(history["val_f1"] * 100)
# plt.legend([f"Treino: {max_train:.2f}", f"Validação: {max_val:.2f}"])
# plt.grid()
# plt.show()

# max_train = np.min(history["loss"].values)
# max_val = np.min(history["val_loss"].values)
# plt.figure(dpi=300)
# plt.title(f"Erro do modelo ao longo das épocas.")
# plt.xlabel("Épocas")
# plt.ylabel("Erro")
# plt.plot(history["loss"] * 100)
# plt.plot(history["val_loss"] * 100)
# plt.legend([f"Treino: {max_train:.4f}", f"Validação: {max_val:.4f}"])
# plt.grid()
# plt.show()

# max_value = np.max(history["bin_acc"].values) * 100
# min_value = np.max(history["val_bin_acc"].values) * 100
# plt.figure(dpi=300)
# plt.title(f"Valor da acurácia binária do modelo ao longo das épocas.")
# plt.xlabel("Épocas")
# plt.ylabel("Acurácia Binária [%]")
# plt.plot(history["bin_acc"] * 100)
# plt.plot(history["val_bin_acc"] * 100)
# plt.legend([f"Treino: {max_value:.2f}", f"Validação: {min_value:.2f}"])
# plt.grid()
# plt.show()

# # %%

# %%

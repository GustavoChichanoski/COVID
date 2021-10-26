from pathlib import Path
from src.models.segmentation.unet_functional import unet_compile, unet_fit, unet_functional
from src.dataset.segmentation.dataset_seg import SegmentationDataset
from src.dataset.segmentation.generator_seg import SegmentationDatasetGenerator

data = Path('D:\Mestrado\datasets\Lung Segmentation')
lung = data / 'CXR_png'
mask = data / 'masks'

dataset = SegmentationDataset(lung, mask)
train, val = dataset.partition(shuffle=False)

params = {'dim': 256}

train_gen = SegmentationDatasetGenerator(
  train[0],
  train[1],
  augmentation=False,
  **params
)
val_gen = SegmentationDatasetGenerator(
  val[0],
  val[1],
  augmentation=False,
  **params
)
model = unet_functional((256,256,1), 32,5,'relu','sigmoid', 1, 0.3)

unet_compile(model)
history = unet_fit(model, train_gen, val_gen)
parar = True
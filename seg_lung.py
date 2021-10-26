# %%
from pathlib import Path

import numpy as np

from src.dataset.segmentation.generator_seg import SegmentationDatasetGenerator as SegDataGen
from src.dataset.segmentation.dataset_seg import SegmentationDataset
from src.models.segmentation.unet import Unet

DIM = 128
BATCH_SIZE = 1
TAMANHO_DATASET = 4

model = Unet(
    depth=6,
    dim=DIM,
    rate=0.5,
    activation='relu',
    final_activation='sigmoid'
)
model.compile(loss='log_cosh_dice', rf=5)
model.build()
model.summary()

data_path = Path('D:\\Mestrado\\datasets\\data_segmentation')
dataset = SegmentationDataset(
    path_lung=data_path / 'lungs',
    path_mask=data_path / 'masks'
)
train, val = dataset.partition(val_size=0.2, tamanho=TAMANHO_DATASET)

params = {
    'batch_size': BATCH_SIZE,
    'dim': DIM,
    'flip_vertical': False,
    'flip_horizontal': False,
    'load_image_in_ram': True
}
train_generator = SegDataGen(train[0],train[1],augmentation=True,**params)
val_generator = SegDataGen(val[0],val[1],augmentation=True,**params)

# model.fit(
#     x=train_generator,
#     validation_data=val_generator,
#     epochs=2
# )

# model.save_weights('D:\\Mestrado\\pesos\')
model.load_weights('D:\\Mestrado\\pesos\\pesos.hdf5')

import matplotlib.pyplot as plt

random_index = np.random.randint(10)
predicts = model.predict(train_generator)

for i in range(len(train_generator)):
    plt.imshow(train_generator[i][0][:][:][0].reshape(DIM,DIM),cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(train_generator[i][1][:][:][0].reshape(DIM,DIM),cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(predicts[i][:][:].reshape(DIM,DIM),cmap='gray')
    plt.axis('off')
    plt.show()
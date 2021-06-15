from pathlib import Path
from src.dataset.generator_seg import SegmentationDataGenerator
from src.dataset.dataset_seg import SegmentationDataset
from src.models.segmentacao.segmentacao_model import Unet

model = Unet()
model.compile()

data_path = Path('D:\Mestrado\data\Lung Segmentation')
dataset = SegmentationDataset(
    path_lung=data_path / 'CXR_png',
    path_mask=data_path / 'masks'
)
train, val = dataset.partition(val_size=0.2, tamanho=10)

train_generator = SegmentationDataGenerator(train[0], train[1], batch_size=1, dim = 256)
val_generator = SegmentationDataGenerator(val[0], val[1], batch_size=1, dim = 256)

model.fit(x=train_generator, validation_data=val_generator)
import cv2 as cv
import numpy as np
from pathlib import Path
from src.data.segmentation.generator_seg import SegmentationDataset
from src.images.read_image import read_images
data_path = Path('D:\\Mestrado\\data\\Lung Segmentation')
new_data_path = Path('D:\\Mestrado\\data_segmentation\\train')

dataset = SegmentationDataset(
    path_lung = data_path / 'CXR_png',
    path_mask = data_path / 'masks'
)

new_data_path_lung = new_data_path / 'lungs'
new_data_path_mask = new_data_path / 'masks'

tamanho_dataset = len(dataset.x)

for (i, path_lung, path_mask) in zip(range(tamanho_dataset), dataset.x, dataset.y):
    print(f'{i} de {tamanho_dataset}')
    lung = read_images(path_lung)
    mask = read_images(path_mask)
    cv.imwrite(f'{new_data_path_lung}/{i:04}.png',lung)
    cv.imwrite(f'{new_data_path_mask}/{i:04}.png',mask)

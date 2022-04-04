import cv2
import numpy as np
import random
import pandas as pd
import albumentations as A
import tensorflow_addons as tfa

from typing import Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from create_csv_of_dataset_segmentation import calculate_parts
from src.images.data_augmentation import (
    add_random_ellipse_brightess, random_brightness, random_contrast,
    random_gamma, random_rotate_image
)
from src.images.data_augmentation import (
    flip_vertical_image, flip_horizontal_image, cut_bot_image, cut_left_image,
    cut_right_image, cut_top_image
)

def augmentation(
    i: int, dataset_path: Path, dataset: Dict, lung: tfa.types.TensorLike,
    mask: tfa.types.TensorLike, pixel_start: Tuple[int, int],
    pixel_end: Tuple[int, int], type: str
) -> Tuple[Dict, int]:
  lung_top = cut_top_image(lung, px_start=pixel_start, px_end=pixel_end)
  lung_bot = cut_bot_image(lung, px_start=pixel_start, px_end=pixel_end)
  lung_left = cut_left_image(lung, px_start=pixel_start, px_end=pixel_end)
  lung_right = cut_right_image(lung, px_start=pixel_start, px_end=pixel_end)
  lung_ellipse = add_random_ellipse_brightess(
      img=lung, px_start=pixel_start, px_end=pixel_end, n_ellipse=2
  )
  lung_brightness = random_brightness(lung)
  lung_contrast = random_contrast(lung)
  lung_gamma = random_gamma(lung)

  dataset = save_image(lung, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_top, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_bot, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_left, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_right, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_ellipse, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_brightness, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_contrast, mask, dataset_path, i, dataset, type)
  i += 1
  dataset = save_image(lung_gamma, mask, dataset_path, i, dataset, type)
  i += 1
  return (dataset, i)


def save_image(
    lung, mask, path: Path, i: int, dataset: dict, type: str
) -> Dict:
  lung_path = path / 'lungs' / '{:04d}.png'.format(i)
  mask_path = path / 'masks' / '{:04d}.png'.format(i)
  dataset['type'][str(i)] = type
  dataset['lung'][str(i)] = str(lung_path)
  dataset['mask'][str(i)] = str(mask_path)
  cv2.imwrite(str(lung_path), lung)
  cv2.imwrite(str(mask_path), mask)
  return dataset


def read_path(path: Path, new_path: Path,  valid: float = 0.2, test: float = 0.2) -> dict:
  if path.is_dir():
    types = ['train', 'valid', 'tests']

    dataset = {}

    dataset['type'] = {}
    dataset['lung'] = {}
    dataset['mask'] = {}

    files = list(path.glob('lungs/*.png'))
    num_files = len(files)
    num_files_by_types = calculate_parts(num_files, test, valid)

    lung_id = 0

    for num_files_of_type, type_file in tqdm(zip(num_files_by_types, types)):
      for file in tqdm(random.sample(files, num_files_of_type)):
        file_mask = convert_lung_mask_path(file)

        dataset['type'][str(lung_id)] = type_file
        dataset['lung'][str(lung_id)] = str(file)
        dataset['mask'][str(lung_id)] = str(file_mask)

        lung = cv2.imread(str(file))
        mask = cv2.imread(str(file_mask))

        lung = cv2.cvtColor(lung, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        lung_id += 1
        if len(lung.shape) > 2:
          y_nonzero, x_nonzero = np.nonzero(mask[:, :, 0])
        else:
          y_nonzero, x_nonzero = np.nonzero(mask[:, :])
        px_start, px_end = (np.min(y_nonzero), np.min(x_nonzero)), \
                            (np.max(y_nonzero), np.max(x_nonzero))

        if type_file != 'tests':
          dataset, lung_id = augmentation(
              i=lung_id,
              dataset_path=new_dataset_path,
              dataset=dataset,
              lung=lung,
              mask=mask,
              pixel_start=px_start,
              pixel_end=px_end,
              type=type_file
          )
        lung_id += 1
        files.remove(file)
    return dataset

def convert_lung_mask_path(file: Path) -> Path:
    file_parts = list(file.parts)
    file_parts[-2] = 'masks'
    file_mask = Path(*file_parts)
    return file_mask

def generate_dict_lungs(path: Path) -> Dict:
  lung_mask_path = {}
  lung_mask_path['lung'] = {}
  lung_mask_path['mask'] = {}

  i = 0

  for file in path.iterdir():
    if file.is_file():
      lung_mask_path['lung'][str(i)] = str(file)
      mask_file = convert_lung_mask_path(file)
      if mask_file.exists():
        lung_mask_path['mask'][str(i)] = mask_file
        i += 1
      else:
        raise ValueError
  return lung_mask_path

dict_lungs = generate_dict_lungs(Path('dataset\lungs'))
df_dataset = pd.DataFrame(dict_lungs)

df_dataset['lung'] = df_dataset['lung'].apply(lambda x: str(Path.cwd() / x))
df_dataset['mask'] = df_dataset['mask'].apply(lambda x: str(Path.cwd() / x))

lungs_path = list(df_dataset['lung'].values)
masks_path = list(df_dataset['mask'].values)

i = 0

dataset_path = Path('dataset')
new_dataset_path = Path('dataset') / 'dataset'
dataset = {}
dataset['type'] = {}
dataset['lung'] = {}
dataset['mask'] = {}

dataset = read_path(dataset_path, new_dataset_path)

df = pd.DataFrame(dataset)
df.to_csv(Path('dataset') / 'dataset' / 'metadata_segmentation_augmentation.csv')

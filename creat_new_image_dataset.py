import cv2
import numpy as np
import random
import pandas as pd
import albumentations as A
import tensorflow_addons as tfa

from typing import Dict
from pathlib import Path
from tqdm import tqdm
from create_csv_of_dataset_segmentation import calculate_parts
import tensorflow_addons as tfa

def save_image(
  lung: tfa.types.TensorLike,
  mask: tfa.types.TensorLike,
  path: Path,
  i: int,
  dataset: dict,
  type: str
) -> Dict:
  lung_path = path / 'lungs' / '{:04d}.png'.format(i)
  mask_path = path / 'masks' / '{:04d}.png'.format(i)
  dataset['type'][str(i)] = type
  dataset['lung'][str(i)] = str(lung_path)
  dataset['mask'][str(i)] = str(mask_path)
  cv2.imwrite(str(lung_path), lung)
  cv2.imwrite(str(mask_path), mask)
  return dataset


def read_path(
  path: Path,
  new_path: Path,
  valid: float = 0.2,
  test: float = 0.2,
  n_copies: int = 8000
) -> Dict:
  if path.is_dir():
    types = ['train', 'valid', 'tests']

    dataset = {}

    dataset['type'] = {}
    dataset['lung'] = {}
    dataset['mask'] = {}

    files = list(path.glob('lungs/*.png'))
    num_files = len(files)
    num_files_by_types = calculate_parts(num_files, test, valid)
    copies_per_image = np.floor(
        n_copies * (num_files_by_types / (num_files * num_files))
    ).astype(int)

    lung_id = 0

    transform_ori = A.Compose(
        [A.InvertImg(always_apply=True),
         A.Equalize(always_apply=True)]
    )
    transform_train = compose_train()
    for num_files_of_type, type_file, copy_per_type in tqdm(
        zip(num_files_by_types, types, copies_per_image)
    ):
      for file in tqdm(random.sample(files, num_files_of_type)):
        file_mask = convert_lung_mask_path(file)

        dataset['type'][str(lung_id)] = type_file
        dataset['lung'][str(lung_id)] = str(file)
        dataset['mask'][str(lung_id)] = str(file_mask)

        lung = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(cv2.imread(str(file_mask)), cv2.COLOR_BGR2GRAY)

        image_ori = transform_ori(image=lung, mask=mask)

        dataset = save_image(
            lung=image_ori['image'],
            mask=image_ori['mask'],
            path=new_path,
            i=lung_id,
            dataset=dataset,
            type=type_file
        )
        lung_id += 1

        if len(lung.shape) > 2:
          y_nonzero, x_nonzero = np.nonzero(mask[:, :, 0])
        else:
          y_nonzero, x_nonzero = np.nonzero(mask[:, :])
        px_start, px_end = (np.min(y_nonzero), np.min(x_nonzero)), \
                           (np.max(y_nonzero), np.max(x_nonzero))

        for _ in range(copy_per_type):
          image = transform_train(
              image=image_ori['image'],
              mask=image_ori['mask']
          )
          dataset = save_image(
              lung=image['image'],
              mask=image['mask'],
              path=new_path,
              i=lung_id,
              dataset=dataset,
              type=type_file
          )
          lung_id += 1
        files.remove(file)
    return dataset


def compose_train() -> A.Compose:
  return A.Compose(
      [
          A.HorizontalFlip(p=0.5),
          A.RandomGamma(
              always_apply=False, p=0.4, gamma_limit=(23, 81), eps=1e-07
          ),
          A.RandomBrightnessContrast(
              always_apply=False,
              p=0.4,
              brightness_limit=(-0.2, 0.2),
              contrast_limit=(-0.2, 0.2),
              brightness_by_max=True
          ),
          A.GaussNoise(var_limit=(200, 300), p=0.4),
          A.ElasticTransform(
              always_apply=False,
              p=0.4,
              alpha=3.0,
              sigma=50.0,
              alpha_affine=50.0,
              interpolation=0,
              border_mode=0,
              value=(0, 0, 0),
              mask_value=None,
              approximate=False
          ),
          A.GridDistortion(
              always_apply=False,
              p=0.4,
              num_steps=5,
              distort_limit=(-0.3, 0.3),
              interpolation=0,
              border_mode=0,
              value=(0, 0, 0),
              mask_value=None
          )
      ]
  )


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

if __name__ == '__main__':

  dict_lungs = generate_dict_lungs(Path('dataset\\lungs'))
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

  dataset = read_path(dataset_path, new_dataset_path, valid=0.2, test=0.1)

  df = pd.DataFrame(dataset)
  df.to_csv(
      Path('dataset') / 'dataset' / 'metadata_segmentation_augmentation.csv'
  )

from pathlib import Path
from typing import Tuple
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(path: str):
  image = tf.io.read_file(path)
  image = tf.image.decode_png(image)
  image = tf.cast(image, tf.float32)
  return image

def resize(image, height: int, width: int):
  image = tf.image.resize(image,
                          [height, width],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return image

def normalize(image):
  norm_image = (image / 127.5) - 1
  return norm_image

def find_mask_image_path(path):
    mask = Path(path)
    mask = list(mask.parts)
    mask[-2] = 'mask'
    mask = Path(*mask)
    return mask

def load_image(path):
  mask_path = find_mask_image_path(path)

  lung = load(path)
  mask = load(mask_path)

  lung = load(path)
  mask = load(mask_path)

  yield lung, mask

def create_dataset(
  df_dataset: pd.DataFrame,
  buffer_size: int,
  batch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Gera o dataset de segmentação

  Args:
      df_dataset (pd.DataFrame): dataframe contendo as imagens
      buffer_size (int): tamanho do buffer de dataset
      batch_size (int): numero de pacotes da imagem

  Returns:
      Tuple[tf.data.Dataset, tf.data.Dataset]: dataset de treino e teste
  """
  df_train = df_dataset.where(df_dataset['type'] == 'train' ).dropna()
  df_tests = df_dataset.where(df_dataset['type'] == 'tests').dropna()

  train_dataset = create_dataset_generator(buffer_size, batch_size, df_train)
  tests_dataset = create_dataset_generator(buffer_size, batch_size, df_tests)

  return train_dataset, tests_dataset

def create_dataset_generator(
  buffer_size: int,
  batch_size: int,
  df_train: pd.DataFrame
) -> tf.data.Dataset:
    df_train_lung_path = df_train.values[:,1]
    train_dataset = tf.data.Dataset.list_files(df_train_lung_path)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset

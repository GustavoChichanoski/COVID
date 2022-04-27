from pathlib import Path
from typing import Tuple
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import cv2


def normalize(
  input_image: tfa.types.TensorLike
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:
  norm_image = tf.cast(input_image, tf.float32) / 255.0
  norm_image -= 1
  return input_image

@tf.function
def load_image(
  x_path: Path,
  y_path: Path,
  dim: int = 256
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

  input_image = tf.io.decode_png(
    x_path,
    channels=0,
    dtype=tf.dtypes.uint8,
    name=None
  )
  input_masks = tf.io.decode_png(
    y_path,
    channels=0,
    dtype=tf.dtypes.uint8,
    name=None
  )

  input_image = tf.image.resize(input_image, (dim,dim))
  input_masks = tf.image.resize(input_masks, (dim,dim))

  input_image = normalize(input_image)

  yield input_image, input_masks

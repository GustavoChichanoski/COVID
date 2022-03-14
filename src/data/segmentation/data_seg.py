from typing import Tuple
import numpy as np
import tensorflow as tf

IMG_SIZE = 256

def parse_image(img_path: str) -> dict:
  """Load an image and its annotation (mask) and returning a dictionary.

  Args:
      img_path (str): Image (not the mask) location

  Returns:
      dict: Dicionary mapping an image and its annotation
  """
  image = tf.io.read_file(img_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.uint8)

  # find the mask path
  # datasets/lungs/0001.png
  # correspond to
  # datasets/masks/0001.png
  mask_path = tf.strings.regex_replace(img_path, "lungs", "masks")
  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_png(mask, channels=1)
  mask = tf.where(mask > 127, np.dtype('uint8').type(0), mask)

  return {'image': image, 'mask': mask}

@tf.function
def normalize(
  image: tf.Tensor,
  mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Rescale the pixel values of the images between -1.0 and 1.0
    compared to [0,255] originally.

  Args:
      image (tf.Tensor): Tensorflow tensor containing an image of size [SIZE,SIZE,3].
      mask (tf.Tensor): Tensorflow tensor containing an image of size [SIZE,SIZE,1].

  Returns:
      Tuple[tf.Tensor, tf.Tensor]: Normalized image and its annotation.
  """
  image = (tf.cast(image, tf.float32) - 127.5) / 127.5
  return image, mask

@tf.function
def load_image_train(datapoint: dict) -> Tuple[tf.Tensor, tf.Tensor]:
  """Apply some transformations to an input dictionary
    containing a train image and its annotation.

  Args:
      datapoint (dict): A dict containing an image and its annotation.

  Returns:
      Tuple[tf.Tensor, tf.Tensor]: A modified image and its annotation.
  """
  image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
  mask = tf.image.resize(datapoint['mask'], (IMG_SIZE, IMG_SIZE))

  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)

  image, mask = normalize(image, mask)
  return image, mask

@tf.function
def load_image_test(datapoint: dict) -> Tuple[tf.Tensor, tf.Tensor]:
  """Normalize and resize a test image and its annotation.

  Notes:
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

  Args:
      datapoint (dict): A dict containing an image and its annotation.

  Returns:
      Tuple[tf.Tensor, tf.Tensor]:  A modified image and its annotation.
  """
  image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
  mask = tf.image.resize(datapoint['mask'], (IMG_SIZE, IMG_SIZE))

  image, mask = normalize(image, mask)

  return image, mask

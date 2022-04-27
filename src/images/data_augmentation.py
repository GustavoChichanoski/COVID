import math
from random import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import cv2
from src.images.process_images import random_pixel
import albumentations as A


def random_rotate_image(
    image: tfa.types.TensorLike, angle: float = 0.0
) -> tfa.types.TensorLike:
  """
        Rotate `image` with tensorflow_addons, based in angle deggres in `angle`, image need be NHWC (number_images,height,width,channels) or HW(height,width)

        Args:
            image (tfa.types.TensorLike): image to be rotated
            angle (float, optional): angle in deggre to rotate image. Defaults to 0.0.

        Returns:
            tfa.types.TensorLike: image rotated
    """
  valid_shape(image, 2, 4)
  rotation = math.radians(angle)
  rotate_image = tfa.image.rotate(image, rotation, interpolation='BILINEAR')
  return rotate_image


def valid_shape(
    image: tfa.types.TensorLike,
    shape_min: int = 2,
    shape_max: int = 4
) -> None:
  """ Valid image to function

        Args:
            image (tfa.types.TensorLike): image to be valid.
            shape_min (int, optional): min length shape of image. Defaults to 2.
            shape_max (int, optional): max length shape of image. Defaults to 4.

        Raises:
            ValueError: if shape image is not valid
    """
  len_image_shape = len(image.shape)
  if len_image_shape != shape_max and len_image_shape != shape_min:
    raise ValueError(
        f'Image must be {shape_min} or {shape_max} shape, not {len_image_shape}: {image.shape}'
    )


def flip_horizontal_image(image: tfa.types.TensorLike) -> tfa.types.TensorLike:
  valid_shape(image, 3, 4)
  return tf.image.flip_left_right(image)


def flip_vertical_image(image: tfa.types.TensorLike) -> tfa.types.TensorLike:
  valid_shape(image, 3, 4)
  return tf.image.flip_up_down(image)

def cut_half(
  image: tfa.types.TensorLike,
  px_start: Tuple[int,int] = (-1,-1),
  px_end: Tuple[int,int] = (-1,-1),
  p: float = 1.0
) -> tfa.types.TensorLike:
  if p < np.random.random():
    return image
  prob = np.random.random_integers(0, 3)
  if prob == 0:
    image = cut_top_image(
        image, px_start=px_start, px_end=px_end
    )
  if prob == 1:
    image = cut_bot_image(
        image, px_start=px_start, px_end=px_end
    )
  if prob == 2:
    image = cut_left_image(
        image, px_start=px_start, px_end=px_end
    )
  if prob == 3:
    image = cut_right_image(
        image, px_start=px_start, px_end=px_end
    )
  return image

def cut_top(
    images: tfa.types.TensorLike, p: float = 1.0
) -> tfa.types.TensorLike:
  if (p < np.random.random()):
    return images
  shape = images.shape
  out = np.array([])
  if len(shape) > 3:
    for image in images:
      image = cut_top_image(image)
      out = np.append(out, image)
  else:
    new_image = cut_top_image(images)
    out = np.append(out, new_image)
  out = np.reshape(out, shape)
  return out


def cut_top_image(
    image: tfa.types.TensorLike,
    px_start: Tuple[int, int] = (-1, -1),
    px_end: Tuple[int, int] = (-1, -1)
) -> tfa.types.TensorLike:
  new_image = image.copy()
  dim = new_image.shape[1]
  dim2 = hort_med(dim=dim, px_start=px_start, px_end=px_end)
  new_image[:dim2, :] = np.zeros((dim2, dim))
  return new_image


def cut_bot(images: tfa.types.TensorLike, p: float = 1.0) -> tfa.types.TensorLike:
  if (p < np.random.random()):
    return images
  out = np.array([])
  shape = images.shape[-2]
  if len(shape) > 3:
    for image in images:
      image = cut_bot_image(image)
      out = np.append(out, image)
  else:
    out = cut_bot_image(images)
  out = np.reshape(out, shape)
  return out


def random_gamma(image: tfa.types.TensorLike) -> tfa.types.TensorLike:
  gamma = normalize(np.random.random_sample(), 0.5, .9)
  return adjust_gamma(image, gamma)


def adjust_gamma(image: tfa.types.TensorLike, gamma: float = 1.0):
  # build a lookup table mapping the pixel values [0, 255] to
  # their adjusted gamma values
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0)**inv_gamma) * 255 for i in np.arange(0, 256)]
                  ).astype("uint8")
  # apply gamma correction using the lookup table
  return cv2.LUT(image, table)


def random_contrast(img: tfa.types.TensorLike) -> tfa.types.TensorLike:
  contrast = normalize(np.random.random_sample(), 0.1, .4) * 255
  return adjust_contrast(img, int(contrast))


def adjust_contrast(
    img: tfa.types.TensorLike, contrast
) -> tfa.types.TensorLike:
  num_channels = 1 if len(img.shape) < 3 else img.shape[-1]
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  f = 131 * (contrast + 127) / (127 * (131 - contrast))
  alpha_c = f
  gamma_c = 127 * (1 - f)
  hsv = cv2.addWeighted(hsv, alpha_c, hsv, 0, gamma_c)
  img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels == 1 else img
  return img


def random_brightness(
    img: tfa.types.TensorLike,
    min_value: float = 0.1,
    max_value: float = 0.3
) -> tfa.types.TensorLike:
  brightness = normalize(np.random.random_sample(), min_value, max_value) * 255
  return adjust_brightness(img, int(brightness))


def normalize(value: float, min_val: float = 0, max_val: float = 1) -> float:
  norm_value = (value * min_val) + (max_val - min_val)
  return norm_value


def adjust_brightness(
    img: tfa.types.TensorLike, value: int
) -> tfa.types.TensorLike:
  num_channels = 1 if len(img.shape) < 3 else img.shape[-1]
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)

  if value >= 0:
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
  else:
    value = int(-value)
    lim = 0 + value
    v[v < lim] = 0
    v[v >= lim] -= value

  final_hsv = cv2.merge((h, s, v))

  img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels == 1 else img
  return img


def cut_bot_image(
    image: tfa.types.TensorLike,
    px_start: Tuple[int, int] = (-1, -1),
    px_end: Tuple[int, int] = (-1, -1)
) -> tfa.types.TensorLike:
  new_image = image.copy()
  dim = new_image.shape[1]
  dim2 = hort_med(dim=dim, px_start=px_start, px_end=px_end)
  new_image[dim2:, :] = np.zeros((1024 - dim2, dim))
  return new_image


def hort_med(
    dim: int, px_start: Tuple[int, int], px_end: Tuple[int, int]
) -> int:
  if px_start[0] != -1:
    return int((px_end[0] + px_start[0]) / 2)
  return int(dim / 2)


def cut_left(images: tfa.types.TensorLike, p: float = 1) -> tfa.types.TensorLike:
  if (p < np.random.random()):
    return images
  out = np.array([])
  shape = images.shape
  if len(shape) > 3:
    for image in images:
      image = cut_left_image(image)
      out = np.append(out, image)
  else:
    out = cut_left_image(images)
  out = np.reshape(out, shape)
  return out


def cut_left_image(
    image: tfa.types.TensorLike,
    px_start: Tuple[int, int] = (-1, -1),
    px_end: Tuple[int, int] = (-1, -1)
) -> tfa.types.TensorLike:
  new_image = image.copy()
  dim = image.shape[1]
  dim2 = med_vert(dim, px_start, px_end)
  new_image[:, :dim2] = np.zeros((dim, dim2))
  return new_image


def cut_right(images: tfa.types.TensorLike, p: float = 1.0) -> tfa.types.TensorLike:
  if (p < np.random.random()):
    return images
  out = np.array([])
  shape = images.shape
  if len(shape) > 3:
    for image in images:
      image = cut_right_image(image)
      out = np.append(out, image)
  else:
    out = cut_right_image(images)
  out = np.reshape(out, shape)
  return out


def cut_right_image(
    image: tfa.types.TensorLike,
    px_start: Tuple[int, int] = (-1, -1),
    px_end: Tuple[int, int] = (-1, -1)
) -> tfa.types.TensorLike:
  new_image = image.copy()
  dim = new_image.shape[1]
  dim2 = med_vert(dim, px_start, px_end)
  new_image[:, dim2:] = np.zeros((dim, 1024 - dim2))
  return new_image


def med_vert(
    dim: int, px_start: Tuple[int, int], px_end: Tuple[int, int]
) -> int:
  if px_start[1] != -1:
    dim2 = int((px_end[1] + px_start[1]) / 2)
  else:
    dim2 = int(dim / 2)
  return dim2


def is_inside_ellipse(
    px: Tuple[int, int], px_center: Tuple[int, int], length: Tuple[int, int]
) -> bool:
  hort = ((px[0] - px_center[0])**2) / (length[0]**2)
  vert = ((px[1] - px_center[1])**2) / (length[1]**2)
  return hort + vert <= 1


def add_random_ellipse_brightess(
    img: tfa.types.TensorLike,
    px_start: Tuple[int, int],
    px_end: Tuple[int, int],
    ellipse_perc: float = 0.05,
    n_ellipse: int = 2
) -> tfa.types.TensorLike:
  dim_img = img.shape[1]
  ellipse_dim = int(img.shape[1] * ellipse_perc)
  length = (ellipse_dim, ellipse_dim)
  new_img = img.copy()
  for _ in range(n_ellipse):
    px_center = random_pixel(px_start, px_end, ellipse_dim)
    new_img = add_ellipse_brightess(new_img, px_center, length, dim_img)
  return new_img


def add_ellipse_brightess(
    img: tfa.types.TensorLike, center: Tuple[int, int], length: Tuple[int, int],
    max_value: int
) -> tfa.types.TensorLike:
  new_img = img.copy()
  bright_img = random_brightness(new_img, 0.1, 0.4)
  y_min_value = center[0] - length[0] if center[0] - length[0] > 0 else 0
  y_max_value = center[0] + length[0] if center[0] + length[0] < max_value else 0
  x_min_value = center[1] - length[1] if center[1] - length[1] > 0 else 0
  x_max_value = center[1] + length[1] if center[1] + length[1] < max_value else 0

  for y in range(y_min_value, y_max_value):
    for x in range(x_min_value, x_max_value):
      if is_inside_ellipse((y, x), center, length):
        new_img[y, x] = bright_img[y, x]
  return new_img


def augmentation(
    batch: tfa.types.TensorLike,
    left_off: bool = False,
    right_off: bool = False,
    top_off: bool = False,
    bot_off: bool = False,
    angle: float = 10.0,
    vertical: bool = False,
    horizontal: bool = False,
) -> tfa.types.TensorLike:
  batch_augmentation = batch
  if angle is not None:
    batch_rotate = random_rotate_image(batch, angle)
    batch_augmentation = np.append(batch_augmentation, batch_rotate, axis=0)
  if vertical:
    batch_flip_vert = tf.image.flip_up_down(batch)
    batch_augmentation = np.append(batch_augmentation, batch_flip_vert, axis=0)
  if horizontal:
    batch_flip_hort = flip_horizontal_image(batch)
    batch_augmentation = np.append(batch_augmentation, batch_flip_hort, axis=0)
  if bot_off:
    batch_cut_bot = cut_bot(batch)
    batch_augmentation = np.append(batch_augmentation, batch_cut_bot, axis=0)
  if top_off:
    batch_cut_top = cut_top(batch)
    batch_augmentation = np.append(batch_augmentation, batch_cut_top, axis=0)
  if right_off:
    batch_cut_right = cut_right(batch)
    batch_augmentation = np.append(batch_augmentation, batch_cut_right, axis=0)
  if left_off:
    batch_cut_left = cut_left(batch)
    batch_augmentation = np.append(batch_augmentation, batch_cut_left, axis=0)
  return batch_augmentation

import math
from typing import Dict, Optional
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

def random_rotate_image(
    image: tfa.types.TensorLike,
    angle: float = 0.0
) -> tfa.types.TensorLike:
    """
        Rotate `image` with tensorflow_addons, based in angle deggres in `angle`, image need be NHWC (number_images,height,width,channels) or HW(height,width)

        Args:
            image (tfa.types.TensorLike): image to be rotated
            angle (float, optional): angle in deggre to rotate image. Defaults to 0.0.

        Returns:
            tfa.types.TensorLike: image rotated
    """
    valid_shape(image,2,4)
    rotation = math.radians(angle)
    rotate_image = tfa.image.rotate(image,rotation,interpolation='BILINEAR')
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
        raise ValueError(f'Image must be {shape_min} or {shape_max} shape, not {len_image_shape}: {image.shape}')

def flip_horizontal_image(
    image: tfa.types.TensorLike
) -> tfa.types.TensorLike:
    valid_shape(image,3,4)
    return tf.image.flip_left_right(image)

def flip_vertical_image(
    image: tfa.types.TensorLike
) -> tfa.types.TensorLike:
    valid_shape(image,3,4)
    return tf.image.flip_up_down(image)

def cut_top(images: tfa.types.TensorLike) -> tfa.types.TensorLike:
    shape = images.shape
    dim = shape[-2]
    out = np.array([])
    if len(shape) > 3:
        for image in images:
            image = cut_top_image(image,dim)
            out = np.append(out,image)
    else:
        image = cut_top_image(images,dim)
        out = np.append(out,image)
    out = np.reshape(out, shape)
    return out

def cut_top_image(image: tfa.types.TensorLike, dim: int) -> tfa.types.TensorLike:
    dim2 = int(dim/2)
    image[:dim2,:,:] = np.zeros((dim2,dim,1))
    return image

def cut_bot(images: tfa.types.TensorLike) -> tfa.types.TensorLike:
    shape = images.shape
    dim = shape[-2]
    out = np.array([])
    if len(shape) > 3:
        for image in images:
            image = cut_bot_image(image,dim)
            out = np.append(out,image)
    else:
        out = cut_bot_image(images,dim)
    out = np.reshape(out, shape)
    return out

def cut_bot_image(image: tfa.types.TensorLike, dim: int) -> tfa.types.TensorLike:
    dim2 = int(dim/2)
    image[dim2:,:,:] = np.zeros((dim2,dim,1))
    return image

def cut_left(images: tfa.types.TensorLike) -> tfa.types.TensorLike:
    shape = images.shape
    dim = int(shape[-2])
    out = np.array([])
    if len(shape) > 3:
        for image in images:
            image = cut_bot_image(image,dim)
            out = np.append(out,image)
    else:
        out = cut_left_image(images,dim)
    out = np.reshape(out, shape)
    return out

def cut_left_image(image: tfa.types.TensorLike, dim: int) -> tfa.types.TensorLike:
    dim2 = int(dim/2)
    image[:,:dim2,:] = np.zeros((dim,dim2,1))
    return image

def cut_right(images: tfa.types.TensorLike) -> tfa.types.TensorLike:
    shape = images.shape
    dim = shape[-2]
    out = np.array([])
    if len(shape) > 3:
        for image in images:
            image = cut_right_image(image,dim)
            out = np.append(out,image)
    else:
        out = cut_right_image(images,dim)
    out = np.reshape(out, shape)
    return out

def cut_right_image(image: tfa.types.TensorLike, dim: int) -> tfa.types.TensorLike:
    dim2 = int(dim/2)
    image[:,dim2:,:] = np.zeros((dim,dim2,1))
    return image

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
        batch_augmentation = np.append(
            batch_augmentation,
            batch_rotate,
            axis=0
        )
    if vertical:
        batch_flip_vert = tf.image.flip_up_down(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_flip_vert,
            axis=0
        )
    if horizontal:
        batch_flip_hort = flip_horizontal_image(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_flip_hort,
            axis=0
        )
    if bot_off:
        batch_cut_bot = cut_bot(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_cut_bot,
            axis=0
        )
    if top_off:
        batch_cut_top = cut_top(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_cut_top,
            axis=0
        )
    if right_off:
        batch_cut_right = cut_right(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_cut_right,
            axis=0
        )
    if left_off:
        batch_cut_left = cut_left(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_cut_left,
            axis=0
        )
    return batch_augmentation
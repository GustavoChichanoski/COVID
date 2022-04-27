"""
    Biblioteca de funções de leitura de imagens pelos caminhos
"""

from pathlib import Path
from typing import List, Union, Any, Tuple, Optional
import cv2 as cv
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf

def read_random_image(
    paths: List[Path],
    id_start: List[int] = [0],
    **params
) -> tfa.types.TensorLike:
    """
        Lê as imagens dos ids contidos em id_start

        Args:
            paths (list): Caminhos completos das imagens a serem lidas.
            id_start (list, optional): ids das imagens a serem lidas.
                                       Defaults to [0,1].

        Returns:
            tfa.types.TensorLike: lista das imagens lidas.
    """
    images = []
    channel = 3 if params['color'] else 1
    shape = (len(id_start), params['dim'], params['dim'], channel)
    for i in id_start:
        image = read_images(images_paths=paths[i],**params)
        images.append(image)
    images = np.array(images)
    images = np.reshape(images, shape)
    return images


def read_sequencial_image(
    paths: List[Path],
    id_start: int = 0,
    id_end: int = 1,
    **params
) -> tfa.types.TensorLike:
    """
        Lê sequencialmente as imagens

        Args:
            paths (list): Caminhos completos das imagens a serem lidas.
            id_start (int, optional): ID inicial da imagem a ser lida.
                                    Defaults to 0.
            id_end (int, optional): ID final da imagem a ser lida.
                                    Defaults to 0.
            normalize (bool, optional): Normaliza a imagem.
                                        Defaults to False.
        Returns:
            list: lista das imagens lidas
    """
    channels = 3 if params['color'] else 1
    shape = (id_end - id_start, params['dim'], params['dim'], channels)
    images = [read_images(paths[i],**params) for i in range(id_start, id_end)]
    images = np.array(images)
    images = np.reshape(images,shape)
    return images

def read_images(
    images_paths: Union[List[Path], Path],
    id_start: Union[List[int], int] = 0,
    id_end: int = -1,
    color: bool = False,
    dim: int = 1024
) -> tfa.types.TensorLike:
    """
        Lê as imagens do listas de caminhos da imagem de start até end -1

        Args:
            images_paths (str or list): Arrays contndo os caminhos das
                                        imagens.
            id_start (int or list,optional): ID do inicio das imagens.
                                             Defaults to 0.
            id_end (int, optional): ID do fim das imagens.
                                    Defaults to -1.
            output_dim (int, optional): Image output dimension.
                                        Defaults to -1.
        
        Returns:
            (np.array or list): retorna uma lista np.array das imagens lidas
    """
    image = None
    params = {'color': color, 'dim': dim}
    if isinstance(images_paths, list):
        if isinstance(id_start, int):
            if id_end < id_start:
                id_end = len(images_paths)
            return read_sequencial_image(images_paths, id_start, id_end, **params)
        return read_random_image(images_paths, id_start, **params)
    if color:
        image = cv.imread(str(images_paths))
    else:
        image = cv.imread(str(images_paths), cv.IMREAD_GRAYSCALE)
    image = resize_image(image,dim,color)
    return image

def resize_image(
    image: tfa.types.TensorLike,
    dim: int,
    color:bool = False
) -> tfa.types.TensorLike:
    """ Resize image to (dim,dim) pass as parameter. The image need be 1 channel with shape:
        - NHWC (num_images,height,width,channels)
        - HW (height,width)
    Args:
        image (tfa.types.TensorLike): image with shape (dim,dim)
        dim (int): dimension of output image
        color (bool): flag to define if image have or not colors

    Returns:
        tfa.types.TensorLike: image resized
    """
    if not color:
        image = tf.expand_dims(image,axis=-1)
    image = tf.image.resize(image,size=[dim,dim])
    return image

def adjust_gamma(
    image: tfa.types.TensorLike,
    gamma: float = 0.5
) -> tfa.types.TensorLike:
    # build a lookup table mapping the pixel values [0, 255] 
    # to their adjusted gamma values
    image = tf.image.adjust_gamma(image,gamma=gamma)
    return image

def read_step(
    images: tfa.types.TensorLike,
    shape: Tuple[int,int,int,int],
    gamma: Optional[float] = 0.5,
    equalize: bool = True
) -> tfa.types.TensorLike:
    dim = shape[1]
    color = shape[-1] != 1
    image_out = np.array([])
    for image in images:
        image = read_images(image,color=color,dim=dim)
        if gamma is not None:
            image = adjust_gamma(image)
        if equalize:
            image = tfa.image.equalize(image)
        image_out = np.append(image_out,image)
    image_out = np.reshape(image_out,shape)
    return image_out

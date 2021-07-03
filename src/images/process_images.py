"""
    Biblioteca referente ao processamento das imagens
"""
from src.images.read_image import read_images
from typing import Optional, Tuple
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Union
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import math

def random_pixel(
    start: Tuple[int,int] = (0, 0),
    end: Tuple[int,int] = (0, 0),
    dim_split: int = 224
) -> Tuple[int, int]:
    """
        Seleciona um pixel randomicamente comecando de start e
        indo end menos a dimensão maxima do corte.

        Args:
            start (tuple, optional): Pixel superior. Defaults to (0,0).
            end (tuple, optional): Pixel inferior. Defaults to (0,0).
            dim_split (int, optional): Dimensão do corte. Defaults to 224.

        Returns:
            (tuple): pixel gerados aleatoriamente
    """
    x_i, y_i = start
    x_e, y_e = end
    pixel_x = np.random.randint(x_i, x_e - dim_split)
    pixel_y = np.random.randint(y_i, y_e - dim_split)
    return pixel_x, pixel_y

def normalize_image(
    images: tfa.types.TensorLike
) -> tfa.types.TensorLike:
    """
        Normaliza as imagens para que todos variem de 0 a 1.

        Args:
            images (list or np.array): Pode ser uma lista de imagens ou uma imagem.

        Returns:
            (np.array): Imagens normalizadas
    """
    return images / 255.0

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

def augmentation_image(
    batch: tfa.types.TensorLike,
    angle: Optional[float] = 5.0,
    flip_horizontal: bool = True,
    flip_vertical: bool = True,
    mean_filter: bool = True
) -> tfa.types.TensorLike:
    batch_augmentation = batch
    if angle is not None:
        batch_rotate = random_rotate_image(batch, angle)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_rotate,
            axis=0
        )
    if flip_vertical:
        batch_flip_vert = tf.image.flip_up_down(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_flip_vert,
            axis=0
        )
    if flip_horizontal:
        batch_flip_hort = flip_horizontal_image(batch)
        batch_augmentation = np.append(
            batch_augmentation,
            batch_flip_hort,
            axis=0
        )
    if mean_filter:
        batch_mean_filter2d = tfa.image.mean_filter2d(batch,padding='SYMMETRIC')
        batch_augmentation = np.append(
            batch_augmentation,
            batch_mean_filter2d,
            axis=0
        )
    return batch_augmentation

# @jit(parallel=True)
def split_images_n_times(
    image: tfa.types.TensorLike,
    n_split: int = 100,
    dim_split: int = 224,
    verbose: bool = True,
    threshold: float = 0.45
) -> tfa.types.TensorLike:
    """
        Recorta a imagem em n_split vezes de tamanhos dim_split ignorando
        recortes totalmente pretos.

        Args:
            image (np.array): imagem a ser recortada
            n_split (int, optional): Numero de cortes. Defaults to 100.
            dim_orig (int, optional): Tamanho da imagem. Defaults to 1024.
            dim_split (int, optional): Tamanho dos cortes. Defaults to 224.

        Returns:
            (tuple): recortes das imagens e o pixel inicial.
    """
    # Criação das listas
    cut_img = np.array([]) # lista de cortes
    cut_pos = np.array([]) # lista de posicoes do corte

    # Define os pixels em que a imgem começa
    y_nonzero, x_nonzero = np.nonzero(image)
    pixel_start, pixel_end = (np.min(y_nonzero), np.min(x_nonzero)), \
                             (np.max(y_nonzero), np.max(x_nonzero))
    shape_cut = (n_split,dim_split,dim_split,1)
    # Cria os n_splits cortes
    iter_n_splits = range(n_split)
    pbar = tqdm(iter_n_splits) if verbose else iter_n_splits
    for _ in pbar:
        # Recebe um corte da imagem não inteiramente preto
        cut, pos = create_non_black_cut(
            image=image,
            start=pixel_start,
            end=pixel_end,
            dim=dim_split,
            threshold=threshold
        )
        cut_norm = normalize_image(cut)
        cut_img = np.append(cut_img, cut_norm) # Armazena o corte
        cut_pos = np.append(cut_pos, pos) # Armaxena o pixel inicial do corte
    cut_img = cut_img.reshape(shape_cut)
    return cut_img, cut_pos


def create_non_black_cut(
    image: tfa.types.TensorLike,
    start: Tuple[int,int] = (0, 0),
    end: Tuple[int,int] = (0, 0),
    dim: int = 224,
    threshold: float = 0.5
) -> tfa.types.TensorLike:
    """
        Cria um recorte que não é totalmente preto

        Args:
            image (np.array): Imagem a ser cortada
            start (tuple, optional): Pixel por onde comecar a cortar.
                                    Defaults to (0, 0).
            end (tuple, optional): Pixel para parar de corte.
                                Defaults to (0, 0).
            dim (int, optional): Dimensão do corte. Defaults to 224.

        Returns:
            numpy.array: recorte da imagem nao totalmente preta
    """
    xi_maior_xf = start[1] > end[1] - dim
    yi_maior_yf = start[0] > end[0] - dim
    offset = 10
    dim_offset = dim + offset
    if xi_maior_xf and yi_maior_yf:
        end = (start[0] + dim_offset, start[1] + dim_offset)
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        return recort, pos
    if xi_maior_xf:
        end = (end[0], start[1] + dim_offset)
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        return recort, pos
    if yi_maior_yf:
        end = (start[0] + dim_offset, end[1])
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        return recort, pos
    pos = random_pixel(start, end, dim)
    recort = create_recort(image, pos, dim)
    valores_validos = np.sum(recort > 0)
    minimo_valores_validos = int(dim * dim * threshold)
    while valores_validos < minimo_valores_validos:
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        valores_validos = np.sum(recort > 0)
    return recort, pos


def create_recort(
    image: tfa.types.TensorLike,
    pos_start: tuple = (0, 0),
    dim_split: int = 224
) -> tfa.types.TensorLike:
    """
        Cria um recorte da imagem indo da posicao inicial até a
        dimensão do recorte

        Args:
            image (np.array): Imagem a ser recortada.
            pos_start (tuple, optional): Posicao do recorte.
                                        Defaults to (0,0).
            dim_split (int, optional): Dimensão do recorte.
                                    Defaults to 224.

        Return:
            (np.array): Recorte da imagem
    """
    pos_end = (pos_start[0] + dim_split, pos_start[1] + dim_split)
    cut = image[pos_start[0] : pos_end[0], pos_start[1] : pos_end[1]]
    return cut

def relu(image: tfa.types.TensorLike) -> tfa.types.TensorLike:
    """
        Retifica a imagem.

        Args:
        -----
            image: imagem a ser retificada. (np.array)

        Returns:
        -------
            (np.array): imagem retificada. (np.array)
    """
    return np.clip(image, 0, None)

def split(
    path_images: Union[List[Path], Path],
    dim: int = 224,
    channels: int = 1,
    n_splits: int = 100,
    threshold: float = 0.35,
    verbose: bool = False,
) -> Union[tfa.types.TensorLike,Tuple[tfa.types.TensorLike,tfa.types.TensorLike]]:
    """
        Return one split of each image in path images to entry in keras model

        Args:
            path_images (List[Path]): list of image to generate splits
            dim (int, optional): Dimensions of cuts. Default to 224.
            channels (int, optional): Number channel of image. Default to 1.
            n_splits (int, optional): Number of cuts. Default to 100.
            threshold (float, optional): Minimum percent of image is valid. Default to 0.35
            verbose (bool, optional): show infos in terminal. Default to False.

        Returns:
            (np.array|tuple(np.array,np.array)): return list of splits images or
                                                    return list of splits images and positions
            
            shape = (batch_size, dim, dim, channels)
    """
    batch_size = len(path_images)
    shape = (batch_size, n_splits, dim, dim, channels)
    images = (read_images(path) for path in path_images)
    split_return = [
        split_images_n_times(
            image,
            n_split=n_splits,
            dim_split=dim,
            verbose=verbose,
            threshold=threshold
        ) for image in images
    ]
    splited_images = split_return[0][0]
    positions = split_return[0][1]
    positions = np.array(positions)
    positions = positions.reshape((batch_size,n_splits, 2))
    cuts = np.array(splited_images)
    cuts_reshape = cuts.reshape(shape)
    return cuts_reshape, positions
"""
    Biblioteca referente ao processamento das imagens
"""
import numpy as np
import tensorflow_addons as tfa
from typing import  Tuple
from tqdm import tqdm

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
    xs, ys = start
    xe, ye = end

    xe -= dim_split
    ye -= dim_split

    if xe < xs:
        aux = xe
        xe = xs
        xs = aux
    if ye < ys:
        aux = ye
        ye = ys
        ys = aux

    pixel_x = np.random.randint(xs, xe)
    pixel_y = np.random.randint(ys, ye)
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
    if len(images.shape) > 3:
        new_images = np.array([])
        for image in images:
            new_images= np.append(new_images, normalize_image(image))
        new_images = np.reshape(new_images,images.shape)
        return new_images
    norm = images / 255
    return norm

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

    if len(image.shape) < 4:
        image = np.reshape(image,
                           (1, image.shape[0], image.shape[1], image.shape[2]))

    # Define os pixels em que a imgem começa
    y_nonzero, x_nonzero = np.nonzero(image[0,:,:,0])
    pixel_start, pixel_end = (np.min(y_nonzero), np.min(x_nonzero)), \
                             (np.max(y_nonzero), np.max(x_nonzero))
    shape_cut = (n_split,dim_split, dim_split, 1)
    shape_pos = (n_split,2)
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
    cut_pos = cut_pos.reshape(shape_pos)
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
    n = 0
    while valores_validos < minimo_valores_validos and n < 500:
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        valores_validos = np.sum(recort > 0)
        n += 1
        if n > 50:
            minimo_valores_validos /= 2
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
    if len(image.shape) < 3:
        cut = image[pos_start[0] : pos_end[0], pos_start[1] : pos_end[1]]
    else:
        cut = image[0,pos_start[0] : pos_end[0], pos_start[1] : pos_end[1], 0]
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
    images: tfa.types.TensorLike,
    dim: int = 224,
    channels: int = 1,
    n_splits: int = 100,
    threshold: float = 0.35,
    verbose: bool = False,
) -> Tuple[tfa.types.TensorLike,tfa.types.TensorLike]:
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
    batch_size = len(images)
    shape = (batch_size, n_splits, dim, dim, channels)
    pos = np.array([])
    cuts = np.array([])
    if verbose:
        pbar = tqdm(total=len(images))
    for image in images:
        splits = split_images_n_times(
            image,
            n_split=n_splits,
            dim_split=dim,
            verbose=verbose,
            threshold=threshold
        )
        splited_images, positions = splits
        pos = np.append(pos,positions)
        cuts = np.append(cuts,splited_images)
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()
    positions = pos.reshape((batch_size,n_splits, 2))
    cuts_reshape = cuts.reshape(shape)

    return cuts_reshape, positions

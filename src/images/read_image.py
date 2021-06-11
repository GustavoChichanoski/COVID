"""
    Biblioteca de funções de leitura de imagens pelos caminhos
"""

from pathlib import Path
from typing import List, Union, Any
import cv2 as cv


def read_random_image(
    paths: list,
    id_start: int = 0,
    color: bool = False
) -> List[Any]:
    """
        Lê as imagens dos ids contidos em id_start

        Args:
            paths (list): Caminhos completos das imagens a serem lidas.
            id_start (list, optional): ids das imagens a serem lidas.
                                       Defaults to [0,1].

        Returns:
            list: lista das imagens lidas.
    """
    images = []
    for i in id_start:
        image = read_images(images_paths=paths[i],color=color)
        images.append(image)
    return images


def read_sequencial_image(
    paths: List[Path],
    id_start: int = 0,
    id_end: int = 1,
    color: bool = False
) -> List[Any]:
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
    images = []
    for i in range(id_start, id_end):
        image = read_images(paths[i],color=color)
        images.append(image)
    return images


def read_images(
    images_paths: Union[List[Path], Path],
    id_start: Union[List[int], int] = 0,
    id_end: int = -1,
    color: bool = False,
    output_dim: int = 1024
):
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
    shape = (output_dim, output_dim)
    image = None
    if isinstance(images_paths, list):
        if isinstance(id_start, int):
            if id_end < id_start:
                id_end = len(images_paths)
            return read_sequencial_image(images_paths, id_start, id_end, color)
        return read_random_image(images_paths, id_start,color)
    if color:
        image = cv.imread(str(images_paths))
    else:
        image = cv.imread(str(images_paths), cv.COLOR_BGR2GRAY)
        image = cv.equalizeHist(image)
    try:
        if output_dim is not None:
            image = cv.resize(image, shape, interpolation=cv.INTER_AREA)
    except:
        raise ValueError(f'O caminho {images_paths} contem erros')
    return image
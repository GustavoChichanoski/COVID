"""
    Biblioteca de funções de leitura de imagens pelos caminhos
"""

from pathlib import Path
from typing import List, Union
import cv2 as cv


def read_random_image(paths: list,
                      id_start) -> list:
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
        image = read_images(iamges_paths=paths[i])
        images.append(image)
    return images


def read_sequencial_image(paths: list,
                          id_start: int = 0,
                          id_end: int = 1) -> list:
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
        image = read_images(paths[i])
        images.append(image)
    return images


def read_images(images_paths: Union[List[Path], Path],
                id_start: Union[List[int], int] = 0,
                id_end: int = -1):
    """
        Lê as imagens do listas de caminhos da imagem de start até end -1

        Args:
            images_paths (str or list): Arrays contndo os caminhos das
                                        imagens.
            id_start (int or list,optional): ID do inicio das imagens.
                                             Defaults to 0.
            id_end (int, optional): ID do fim das imagens.
                                    Defaults to -1.
            normalize (bool, optional): A saída sera normalizada
                                              Default to false.
        Returns:
            (np.array or list): retorna uma lista np.array das imagens lidas
    """
    if isinstance(images_paths, list):
        if isinstance(id_start, int):
            if id_end < id_start:
                id_end = len(images_paths)
            return read_sequencial_image(images_paths, id_start, id_end)
        return read_random_image(images_paths, id_start)
    image = cv.imread(str(images_paths),0)
    image = equalize_histogram(image)
    return image

def equalize_histogram(image):
    return cv.equalizeHist(image)
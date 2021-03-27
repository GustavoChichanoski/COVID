"""
    Biblioteca de funções de leitura de imagens pelos caminhos
"""

import cv2 as cv


def read_random_image(paths: list,
                      id_start: list) -> list:
    """
        Lê as imagens dos ids contidos em id_start

        Args:
            paths (list): Caminhos completos das imagens a serem lidas.
            id_start (list, optional): ids das imagens a serem lidas.
                                       Defaults to [0,1].

        Returns:
            list: lista das imagens lidas.
    """
    if images is None:
        images = []
    for i in id_start:
        images.append(read_images(paths[i]))
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

        Returns:
            list: lista das imagens lidas
    """
    images = []
    for i in range(id_start, id_end):
        images.append(read_images(paths[i]))
    return images


def read_images(images_paths,
                id_start=0,
                id_end:int = -1):
    """
        Lê as imagens do listas de caminhos da imagem de start até end -1

        Args:
            images_paths (str or list): Arrays contndo os caminhos das
                                        imagens.
            id_start (int or list,optional): ID do inicio das imagens.
                                             Defaults to 0.
            id_end (int, optional): ID do fim das imagens.
                                    Defaults to -1.

        Returns:
            (np.array or list): retorna uma lista np.array das imagens lidas
    """
    if isinstance(images_paths, list):
        if isinstance(id_start, int):
            if id_end < id_start:
                id_end = len(images_paths)
            return read_sequencial_image(images_paths, id_start, id_end)
        return read_random_image(images_paths, id_start)
    return cv.imread(images_paths)

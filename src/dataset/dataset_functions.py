from typing import List
import numpy as np
from os.path import join
from os import listdir
from pathlib import Path

def normalize_image(x):
    """
        normaliza uma imagem de -1 a 1
    """
    x_max = np.max(x)
    x_min = np.min(x)
    x_new = (x - x_min) / (x_max - x_min)
    x_normalizado = (x_new * 2) - 1
    return x_normalizado


def listdir_full(path: str) -> list:
    """ É um os.listdir só que retornando todo o caminho do arquivo.
    ex: listdir_full_path('img')
        return ['img/0.png','img/1.png]
    Args:
        path (str): caminho pai dos arquivos

    Returns:
        (list): lista de strings contendo o caminho todo das imagens.
    """
    filenames = listdir(path)
    paths = [path / filename for filename in filenames]
    return paths


def zeros(len_array: int) -> List[float]:
    """ Gera uma lista de zeros de tamanho len_array

    Args:
        len_array (int): numero de zeros da lista.

    Returns:
        list: lista de zeros com tamanho len_array
    """
    array = []
    for _ in range(len_array):
        array.append(0)
    return array

def get_folders_names(path: str) -> List[List[str]]:
    """
        Retorna o nomes das pastas na pasta de treino.
        Args:
            path (str): Caminho a ser analizado
        Returns:
            (list): Retorna o nome das pastas.
    """
    folder_names = listdir(path)
    return sorted(folder_names)


def proportion_of_files_in_folder(folder_names: List[str],
                                  files_in_folder) -> List[float]:
    """
        Retorna a proporção dos arquivos entre os folders
        Args:
            folder_names (List[str]):
                Lista contendo os nomes das pastas onde os dados estão listados
            files_in_folder (List[int]):
                Numero de arquivos por pasta
        Returns:
            (list): proporções de 0 a 1
    """
    prop = np.zeros((1,len(folder_names)))
    total = 0
    files = files_in_folder
    for index, folder in enumerate(files):
        total += len(folder)
    for index, folder in enumerate(files):
        prop[index] = len(folder)/total
    return prop

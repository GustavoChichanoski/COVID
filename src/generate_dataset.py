import numpy as np
import os
from read_image import read_images as ri
from process_images import split_images_n_times as split

def listdir_full_path(path='./data/Covid/0000.png'):
    """ É um os.listdir só que retornando todo o caminho do arquivo.

    Args:
        path (str): caminho pai dos arquivos

    Returns:
        (list): lista de strings contendo o caminho todo das imagens.
    """
    urls = os.listdir(path)
    full_path = [os.path.join(path, url) for url in urls]
    return full_path

def proportion_class(path_type: str) -> list:
    """[summary]

    Args:
        path_type (str): O caminho onde as imagens estão separadas em classes.

    Returns:
        list: Retorna uma lista contendo a proporção de imagens no dataset.
    """
    diseases = [len(os.listdir(disease))
                for disease in listdir_full_path(path_type)]
    total = 0
    for disease in diseases:
        total += disease
    proportion = np.array(diseases)/total
    return proportion


def number_images_load_per_step(path_type: str, img_loads=10) -> list:
    """Retorna a proporção de imagens que devem ser carregadas por classes a cada passo.

    Args:
        path_type (str): caminho contendo as classes
        img_loads (int, optional): Número de imagens a ser carregadas por passo. Defaults to 10.

    Returns:
        list: numero de imagens que devem ser carregadas por classe.
    """
    proportion = proportion_class(path_type)
    img_ready_load = img_loads
    img_per_class = []
    for p_class in proportion:
        img_per_class.append(np.floor(p_class*img_loads))
        img_ready_load -= img_per_class[-1]
    img_per_class[-1] += img_ready_load
    return img_per_class

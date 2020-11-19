import os
import numpy as np
from read_image import read_images as ri
from process_images import split_images_n_times as split
from os.path import join

K_SPLIT = 100
DIM_ORIG = 1024
DIM_SPLIT = 224


def listdir_full(path: str) -> list:
    """ É um os.listdir só que retornando todo o caminho do arquivo.
    ex: listdir_full_path('img')
        return ['img/0.png','img/1.png]
    Args:
        path (str): caminho pai dos arquivos

    Returns:
        (list): lista de strings contendo o caminho todo das imagens.
    """
    urls = os.listdir(path)
    full_path = [os.path.join(path, url) for url in urls]
    return full_path


def proportion_class(path_type: str) -> list:
    """ Calcula as proporções entre as classes

        Args:
            path_type (str): O caminho onde as imagens estão separadas em classes.

        Returns:
            list: Retorna uma lista contendo a proporção de imagens no dataset.
    """
    path_diseases = listdir_full(path_type)
    path_diseases_sorteds = sorted(path_diseases)
    diseases = [len(os.listdir(disease))
                for disease in path_diseases_sorteds]
    total = 0
    for len_disease in diseases:
        total += len_disease
    proportion = np.array(diseases)/total
    return proportion


def img_diseases_step(path_type: str,
                      img_loads: int = 10) -> list:
    """ Retorna a proporção de imagens que devem ser carregadas
        por classes a cada passo.

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
        per_img = int(np.floor(p_class*img_loads))
        img_per_class.append(per_img)
        img_ready_load -= img_per_class[-1]
    img_per_class[-1] += img_ready_load
    return img_per_class


def step_x(path: str,
           start: int = 0,
           steps: int = 1,
           disease: str = 'Covid') -> tuple:
    path_covid = join(path, disease)
    files_covid = listdir_full(path_covid)
    end = start + steps
    images = ri(files_covid,
                start,
                end)
    return images


def step_y(steps: int = 1,
           type_disease: str = 'Covid',
           step_disease: int = 1) -> list:
    """ Retorna o dataset Y

    Args:
        steps (int, optional): Quantidade de imagens carregadas. Defaults to 1.
        type_disease (str, optional): Tipo de doença. Defaults to 'Covid'.
        step_disease (int, optional): Número de cortes na imagem. Defaults to 1.

    Returns:
        list: Saída para o keras
    """
    ouput_disease = []
    for _ in range(steps):
        step_ouput = disease_of_cuts(type_disease,
                                     step_disease)
        ouput_disease.append(step_ouput)
    return ouput_disease


def create_dataset(path: str,
                   id_class: list = [0, 0, 0],
                   img_per_steps: int = 10,
                   dim_orig: int = DIM_ORIG,
                   dim_split: int = DIM_SPLIT,
                   n_splits: int = K_SPLIT):
    """ Cria o dataset do passo

        Args:
            path (str): Caminho do dataset. Defaults to [0,0,0].
            id_class (list): Id inicial das imagens.
            img_per_steps (int, optional): Imagens por passo. Defaults to 10.
            dim_orig (int, optional): Dimnesao original da imagem. Defaults to DIM_ORIG.
            dim_split (int, optional): Dimensao dos cortes. Defaults to DIM_SPLIT.
            n_splits (int, optional): Numero de cortes. Defaults to K_SPLIT.

        Returns:
            (tuple): Retorna o x e y do dataset.
    """
    step = img_diseases_step(path,
                             img_per_steps)
    dataset_y = []
    pos_x = []
    dataset_x = []
    diseases = sorted(os.listdir)
    for i in len(diseases):
        images = step_x(path, id_class[i],
                        step[i],
                        diseases[i])
        cuts, pos = split(images,
                          n_splits,
                          dim_split,
                          dim_orig)
        pos_x.append(pos)
        dataset_x.append(cuts)
    return dataset_x, pos_x, dataset_y


def disease_of_cuts(disease_cut: str = 'Covid',
                    n_splis: int = K_SPLIT) -> list:
    """
        Retorna a saída esperada para todos da Deep Learning

        Args:
            disease_cut (str, optional): Doença dos cortes. Defaults to 'Covid'.
            n_splis (int, optional): Numero de cortes. Defaults to 100.

        Returns:
            (tuple): Dataset para ser passado para o keras
    """
    if diseases is None:
        diseases = []
    for _ in range(n_splis):
        diseases.append(disease_Y(disease_cut))
    return diseases


def disease_Y(path: str) -> list:
    """
        Retorna a saída para o kera

        Args:
            path (str): Doença da imagem

        Returns:
            (list): lista da saída do modelo desejada
    """
    if path == 'Covid':
        return [1, 0, 0]
    elif path == 'Pneumonia':
        return [0, 0, 1]
    else:
        return [0, 1, 0]

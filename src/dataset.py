"""
    Gera o dataset.
"""
from os.path import join
from os import listdir
from read_image import read_images as ri
from process_images import create_non_black_cut as cut
from process_images import split_images_n_times as splits
import numpy as np


class Dataset:
    """
        Cria o dataset para o keras.
    """

    def __init__(self,
                 path_data: str,
                 number_splits: int = 100,
                 files_per_steps: int = 10,
                 dimension_original: int = 1024,
                 dimension_cut: int = 224,
                 channels: int = 3):
        """
            Args:
                path_data (str): Caminho onde se encontra os dados dos raios-x
                number_splits (int): numero de cortes por imagem.
                dimension_original (int): dimensão da imagem original
                dimension_cut (int): dimensão dos recortes
        """
        self.path_data = path_data
        self.number_cuts = number_splits
        self.dimension_cut = dimension_cut
        self.dimension_original = dimension_original
        self.files_per_steps = files_per_steps
        self.channels = 3

        self.path_train = join(path_data, 'train')
        self.path_test = join(path_data, 'test')

        self.folder_names = self.get_folders_names()
        self.files_in_folder = self.get_files_in_folder()

        self.ids = zeros(len(self.folder_names))

    def get_features_per_steps(self):
        """
            Gera as features para ser inseridas no modelo do Keras.
            Returns:
                (list): imagens de entrada nos modelos.
        """
        features = []
        n_files = self.n_files_per_step()
        for index, folder in enumerate(self.folder_names):
            ids = self.ids[index]
            paths = self.files_in_folder[index]
            full = join(self.path_train, folder)
            full = [join(full, path) for path in paths]
            end = ids + n_files[index]
            imgs = ri(full, ids, end)
            for img in imgs:
                recorts = splits(img,
                                 self.number_cuts,
                                 self.dimension_original,
                                 self.dimension_cut)
                recorts = np.array(recorts)
                recorts = recorts.reshape((self.number_cuts,
                                           self.dimension_original,
                                           self.dimension_cut,
                                           self.channels))
            features.append(recorts)
            self.ids[index] = end
        features = np.array(features)
        features = features.reshape((self.number_cuts*self.files_per_steps,
                                     self.dimension_original,
                                     self.dimension_cut,
                                     self.channels))
        return features

    def get_output(self):
        """ Gera as saídas dos Keras

            Returns:
                (list): saidas para o Keras
        """
        n_files = self.n_files_per_step()
        folders = self.folder_names
        outputs = []
        for index in range(len(folders)):
            output = zeros(len(folders))
            output[index] = 1
            outputs.append(output
                           * n_files[index]
                           * self.number_cuts)
        return outputs

    def n_files_per_step(self):
        """
            Retorna o numero de arquivos que devem ser lidos por passo.
            Returns:
                (list): numero de arquivos por pasta
        """
        proportion = self.proportion_of_files_in_folder()
        for index, _ in enumerate(proportion):
            proportion[index] *= self.files_per_steps
            proportion[index] = int(proportion[index])
        proportion[-1] += (self.files_per_steps - sum(proportion))
        return proportion

    def get_folders_names(self):
        """
            Retorna o nomes das pastas na pasta de treino.
            Returns:
                (list): Retorna o nome das pastas.
        """
        folder_names = listdir(self.path_train)
        return sorted(folder_names)

    def get_files_in_folder(self):
        """
            Retorna o nomes dos arquivos contidos nas pastas.
            Returns:
                (list): nomes dos arquivos nas pastas
        """
        n_files = []
        for folder in self.folder_names:
            full = join(self.path_train, folder)
            n_files.append(listdir(full))
        return n_files

    def proportion_of_files_in_folder(self):
        """
            Retorna a proporção dos arquivos entre os folders
            Returns:
                (list): proporções de 0 a 1
        """
        prop = zeros(len(self.folder_names))
        total = 0
        files = self.files_in_folder
        for index, folder in enumerate(files):
            total += len(folder)
        for index, folder in enumerate(files):
            prop[index] = len(folder)/total
        return prop


def zeros(len_array: int) -> list:
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


def listdir_full(path: str) -> list:
    """ É um os.listdir só que retornando todo o caminho do arquivo.
    ex: listdir_full_path('img')
        return ['img/0.png','img/1.png]
    Args:
        path (str): caminho pai dos arquivos

    Returns:
        (list): lista de strings contendo o caminho todo das imagens.
    """
    urls = listdir(path)
    full_path = [join(path, url) for url in urls]
    return full_path

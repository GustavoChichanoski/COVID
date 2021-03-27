"""
    Gera o dataset.
"""
from os.path import join
from os import listdir
from src.dataset.dataset_functions import get_folders_names, listdir_full, proportion_of_files_in_folder, zeros
from typing import List
from src.images.read_image import read_images as ri
from src.images.process_images import split_images, split_images_n_times as splits
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    """
        Cria o dataset para o keras.
    """

    def __init__(self,
                 path_data: str,
                 dimension_original: int = 1024,
                 dimension_cut: int = 224,
                 channels: int = 3,
                 train: bool = True):
        """
            Args:
                path_data (str): Caminho onde se encontra os dados dos raios-x
                number_splits (int): numero de cortes por imagem.
                dimension_original (int): dimensão da imagem original
                dimension_cut (int): dimensão dos recortes
        """
        self.train = train
        self.path_data = path_data
        self.dimension_cut = dimension_cut
        self.dimension_original = dimension_original
        self.channels = channels
        self.path_train = join(path_data, 'train')
        self.path_test = join(path_data, 'test')
        self.folder_names = get_folders_names(path=self.path_train)
        self.files_in_folder = self.get_files_in_folder()
        self.ids = zeros(len(self.folder_names))
        self.x = self.input()
        self.y = self.output()

    def get_filenames(self):
        if self.train:
            return [listdir_full(join(self.path_train, label))
                    for label in self.folder_names]
        else:
            return [listdir_full(join(self.path_test, label))
                    for label in self.folder_names]

    def partition(self,
                  val_size: float = 0.2):
        """ Retorna a entrada e saidas dos keras.

            Args:
            -----
                val_size (float, optional): Define o tamanho da validacao.
                                            Defaults to 0.2.
            Returns:
            --------
                (test), (val), pos: Saida para o keras.
        """
        # t : train - v : validation
        t_in, v_in, t_out, v_out = train_test_split(self.x,
                                                    self.y,
                                                    test_size=val_size,
                                                    shuffle=True,
                                                    random_state=42)
        train, val = (t_in, t_out), (v_in, v_out)
        return train, val

    def input(self):
        if self.train:
            path = self.path_train
        else:
            path = self.path_test
        paths = []
        for i in range(len(listdir(path))):
            paths.extend(self.files_in_folder[i])
        return paths

    def output(self) -> List[List[str]]:
        output = np.eye(len(self.folder_names))
        outputs = []
        total = 0
        for i in range(len(self.folder_names)):
            total += len(self.files_in_folder[i])
            out = [output[i]] * len(self.files_in_folder[i])
            outputs = np.append(outputs, out)
        outputs = np.array(outputs)
        outputs = outputs.reshape(len(self.x), len(self.folder_names))
        return outputs

    def get_files_in_folder(self) -> List[int]:
        """
            Retorna o nomes dos arquivos contidos nas pastas.
            Returns:
                (list): nomes dos arquivos nas pastas
        """
        files_per_folder = []
        for folder in self.folder_names:
            if self.train:
                path = join(self.path_train, folder)
            else:
                path = join(self.path_test, folder)
            files_per_folder.append(listdir_full(path))
        return files_per_folder

    def custom_prop(self, value: List[float] = [1/3, 1/3]) -> None:
        if len(value) == len(self.folder_names) - 1:
            total = 0
            for valor in value:
                total += valor
            value.append(1 - total)
        self.proportion = value
        return None

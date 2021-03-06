"""
    Gera o dataset.
"""
from os.path import join
from os import listdir
from src.images.read_image import read_images as ri
from src.images.process_images import split_images_n_times as splits
import numpy as np
from sklearn.model_selection import train_test_split


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
        self.channels = channels

        self.path_train = join(path_data, 'train')
        self.path_test = join(path_data, 'test')

        self.folder_names = self.get_folders_names()
        self.files_in_folder = self.get_files_in_folder()

        self.ids = zeros(len(self.folder_names))

    def reset_ids(self):
        """ Reset os ids
        """
        self.ids = zeros(len(self.ids))

    def step(self, val_size=0.2):
        """ Retorna a entrada e saidas dos keras.

            Args:
                val_size (float, optional): Define o tamanho da validacao. Defaults to 0.2.

            Returns:
                [type]: Saida para o keras.
        """
        outputs = self.get_step_output()
        features, pos = self.get_features_per_steps()
        # t : train - v : validation
        t_in, v_in, t_out, v_out = train_test_split(features,
                                                    outputs,
                                                    test_size=val_size,
                                                    shuffle=True,
                                                    random_state=42)
        return (t_in, t_out), (v_in, v_out), pos

    def get_features_per_steps(self):
        """
            Gera as features para ser inseridas no modelo do Keras.
            Returns:
                (list): imagens de entrada nos modelos.
        """
        features = []
        pos = []
        number_files_per_folder = self.number_files_in_step()
        for index, folder in enumerate(self.folder_names):
            ids = self.ids[index]
            paths = self.files_in_folder[index]
            full = join(self.path_train, folder)
            full = [join(full, path) for path in paths]
            end = ids + number_files_per_folder[index]
            imgs = ri(full, ids, end)
            for img in imgs:
                recorts, pos = splits(img,
                                      self.number_cuts,
                                      self.dimension_original,
                                      self.dimension_cut)
                features.append(recorts)
                pos.append(pos)
            self.ids[index] = end
        features = np.array(features)
        features = features.reshape((self.number_cuts*self.files_per_steps,
                                     self.dimension_cut,
                                     self.dimension_cut,
                                     self.channels))
        return features, pos

    def get_output(self, folder):
        """ Retorna a saída a ser inserida no Keras.
            Ex: [0,0,1]

            Args:
                folder (str): nome da pasta

            Returns:
                (list): retorna a saída com base na pasta
        """
        number_folders = len(self.folder_names)
        output = zeros(number_folders)
        index = self.folder_names.index(folder)
        output[index] = 1
        return output

    def get_step_output(self):
        """ Gera as saídas dos Keras

            Returns:
                (list): saidas para o Keras
        """
        number_files_per_folder = self.number_files_in_step()
        folders = self.folder_names
        outputs = []
        for index, folder in enumerate(folders):
            output = self.get_output(folder)
            files_to_load = number_files_per_folder[index]
            output = [output] * files_to_load * self.number_cuts
            outputs.extend(output)
        outputs = np.array(outputs)
        outputs = outputs.reshape((self.number_cuts
                                   * self.files_per_steps,
                                   len(folders)))
        return outputs

    def number_files_in_step(self):
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
        number_files_per_folder = []
        for folder in self.folder_names:
            full = join(self.path_train, folder)
            number_files_per_folder.append(listdir(full))
        return number_files_per_folder

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

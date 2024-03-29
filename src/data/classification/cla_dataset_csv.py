"""
    Gera o dataset.
"""
import pandas as pd
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass, field
import tensorflow_addons


@dataclass
class DatasetCsv:
    """ Cria o dataset para o keras. """
    dataset: pd.DataFrame
    dimension_original: int = 1024
    dimension_cut: int = 224
    channels: int = 3
    train: bool = True
    column_x: str = 'segmentation'
    column_y: str = 'type'
    labels_names: Optional[List[str]] = None
    """
        Args:
            path_data (str): Caminho onde se encontra os dados dos raios-x
            number_splits (int): numero de cortes por imagem.
            dimension_original (int): dimensão da imagem original
            dimension_cut (int): dimensão dos recortes
    """
    _lazy_label_names: Optional[List[Path]] = None
    _lazy_files_in_folder: Optional[List[Path]] = None
    _lazy_x: Optional[List[Path]] = None
    _lazy_y: Optional[Any] = None

    @property
    def files_in_folder(self) -> List[int]:
        """
            Retorna o nomes dos arquivos contidos nas pastas.
                Returns:
                    (list): nomes dos arquivos nas pastas
        """
        if self._lazy_files_in_folder is None:
            self._lazy_files_in_folder = [
                list(folder.iterdir()) for folder in self.label_names
            ]
        return self._lazy_files_in_folder

    @property
    def number_files_in_folders(self) -> List[Union[List[int], int]]:
        """ The number of files in each folders.

            Examples:
            a
            |__ b
            |   |_ d.png
            |
            |__ c
            |   |_ e.png
            |
            |__ d

            >>> number_files_in_fodler(Path(a))
            >>> [[1],[1],[0]]

            Returns:
                np.array: number files in each folder
        """
        if self._lazy_number_files_in_folders is None:
            files = np.array([
                len(folder) for folder in self.files_in_folder
            ])
            self._lazy_number_files_in_folders = files
        return self._lazy_number_files_in_folders

    # @property
    # def label_names(self) -> List[Path]:
    #     """
    #         Name of classes base in the last folders before images

    #         Returns:
    #             List[Path]: [description]
    #     """
    #     if self._lazy_label_names is None:
    #         self._lazy_label_names = sorted(self.labels)
    #     return self._lazy_label_names

    @property
    def y(self) -> tensorflow_addons.types.TensorLike:
        """
            Generate the y values of inputs images based in your class

            Returns:
                numpy.array: the classes of images
        """
        if self._lazy_y is None:
            # Recebe os nomes dos rotulos
            labels = np.array(list(self.label_names))
            # Acha o tamanho dos rotulos
            len_labels = len(labels)
            # Cria a matriz dos resultados de saída
            label_eyes = np.eye(len_labels)
            # Criacao vetor de saída
            outputs = np.array([])
            # Preenchimento do vetor de saídas
            for x_label in self.dataset[self.column_y].values:
                # label verdadeiro
                # Acha o index da label
                for i in range(len_labels):
                    if labels[i] in x_label:
                        break
                out = label_eyes[i]
                outputs = np.append(outputs, out)
            outputs = outputs.reshape(len(self.x), len_labels)
            self._lazy_y = outputs
        return self._lazy_y

    @property
    def x(self) -> List[Path]:
        if self._lazy_x is None:
            self._lazy_x = self.dataset[self.column_x].values
        return self._lazy_x

    def set_labels(
        self,
        labels: List[str] = None
    ) -> None:
        if labels is None:
            labels = ['Covid', 'Normal', 'Pneumonia']
        self.label_names = labels

    def calcular_tamanhos_datasets(
        self,
        tamanho_total: int,
        test_size: float = 0.1,
        validation_size: float = 0.2,
    ) -> List[int]:
        tests_dataset = int(tamanho_total * test_size)
        valid_datatet = int((tamanho_total - tests_dataset) * validation_size)
        train_dataset = tamanho_total - tests_dataset - valid_datatet
        return [train_dataset, valid_datatet, tests_dataset]

    def partition(
        self,
        val_size: float = 0.2,
        test_size: float = 0.1,
        tamanho: int = 0,
        shuffle: bool = True
    ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        """ Retorna a entrada e saidas dos keras.

            Args:
            -----
                val_size (float, optional): Define o tamanho da validacao.
                                            Defaults to 0.2.
                tamanho (int, optional):
                    Tamanho maximo do dataset a ser particionado.
                    Default to 0.
                shuffle (bool, optional):
                    Embaralhar os valores.
                    Default to True

            Returns:
            --------
                (train), (val): Saida para o keras.
        """
        # t : train - v : validation
        tam_max = self.tamanho_maximo(tamanho)
        x, y = self.x[:tam_max], self.y[:tam_max]

        train_in, tests_in, train_out, tests_out = train_test_split(
            x, y, test_size=test_size, shuffle=shuffle
        )
        train_in, val_in, train_out, val_out = train_test_split(
            train_in, train_out, test_size=val_size, shuffle=shuffle
        )

        train = (train_in, train_out)
        val = (val_in, val_out)
        test = (tests_in, tests_out)

        return train, val, test

    def tamanho_maximo(self, tamanho: int) -> int:
        if tamanho == 0:
            return len(self.x)
        if tamanho < len(self.x) + 1:
            return tamanho
        return len(self.x)

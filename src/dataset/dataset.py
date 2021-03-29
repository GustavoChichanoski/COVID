"""
    Gera o dataset.
"""
from os.path import join
from os import listdir
from typing import Any, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass, fields


@dataclass
class Dataset:
    """
        Cria o dataset para o keras.
    """
    path_data: Path
    dimension_original: int = 1024
    dimension_cut: int = 224
    channels: int = 3

    _lazy_label_names: Optional[List[Path]] = None
    _lazy_files_in_folder: Optional[List[Path]] = None
    _lazy_x: Optional[List[Path]] = None
    _lazy_y: Optional[Any] = None
    """
        Args:
            path_data (str): Caminho onde se encontra os dados dos raios-x
            number_splits (int): numero de cortes por imagem.
            dimension_original (int): dimensão da imagem original
            dimension_cut (int): dimensão dos recortes
    """

    @property
    def files_in_folder(self):
        """
            Retorna o nomes dos arquivos contidos nas pastas.
            Returns:
                (list): nomes dos arquivos nas pastas
        """
        if self._lazy_files_in_folder is None:
            self._lazy_files_in_folder = [list(folder.iterdir()) for folder in self.label_names]
        return self._lazy_files_in_folder

    @property
    def label_names(self) -> List[Path]:
        if self._lazy_label_names is None:
            folder_names = self.path_data.iterdir()
            self._lazy_label_names = sorted(folder_names)
        return self._lazy_label_names

    @property
    def y(self) -> Any:
        """Retorna 

        Returns:
            numpy.array:
        """
        if self._lazy_y is None:
            labels = list(self.label_names)
            len_labels = len(labels)
            label_eyes = np.eye(len_labels)
            files = list(self.files_in_folder)
            outputs = []
            for label_eye, file in zip(label_eyes, files):
                out = [label_eye] * len(file)
                outputs = np.append(outputs, out)
            outputs = np.array(outputs)
            self._lazy_y = outputs.reshape(len(self.x), len_labels)
        return self._lazy_y

    @property
    def x(self) -> List[Path]:
        if self._lazy_x is None:
            self._lazy_x = sum(list(self.files_in_folder), [])
        return self._lazy_x

    def partition(self,
                  val_size: float = 0.2) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        """ Retorna a entrada e saidas dos keras.

            Args:
            -----
                val_size (float, optional): Define o tamanho da validacao.
                                            Defaults to 0.2.
            Returns:
            --------
                (test), (val): Saida para o keras.
        """
        # t : train - v : validation
        train_in, val_in, train_out, val_out = train_test_split(
            self.x,
            self.y,
            test_size=val_size,
            shuffle=True
        )
        train, val = (train_in, train_out), (val_in, val_out)
        return train, val

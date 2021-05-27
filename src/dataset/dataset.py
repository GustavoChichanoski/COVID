"""
    Gera o dataset.
"""
from os.path import join
from os import listdir
from typing import Any, List, Optional, Tuple, Union
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
    train: bool = True

    _lazy_label_names: Optional[List[Path]] = None
    _lazy_files_in_folder: Optional[List[Path]] = None
    _lazy_x: Optional[List[Path]] = None
    _lazy_y: Optional[Any] = None
    _lazy_number_files_in_folders: Optional[List[str]] = None
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
            self._lazy_files_in_folder = [
                list(folder.iterdir()) for folder in self.label_names
            ]
        return self._lazy_files_in_folder

    @property
    def number_files_in_folders(self):
        if self._lazy_number_files_in_folders is None:
            files = np.array([
                len(folder) for folder in self.files_in_folder
            ])
            self._lazy_number_files_in_folders = files
        return self._lazy_number_files_in_folders

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
            # Recebe os nomes dos rotulos
            labels = np.array(list(self.label_names))
            # Acha o tamanho dos rotulos
            len_labels = len(labels)
            # Cria a matriz dos resultados de saída
            label_eyes = np.eye(len_labels)
            # Cricao vetor de saída
            outputs = np.array([])
            # Preenchimento do vetor de saídas
            for x in self.x:
                # label verdadeiro
                x_label = x.parts[-2]
                # Acha o index da label
                for i in range(len_labels):
                    if x_label == labels[i].name:
                        break
                out = label_eyes[i]
                outputs = np.append(outputs,out)
            outputs = outputs.reshape(len(self.x), len_labels)
            self._lazy_y = outputs
        return self._lazy_y

    @property
    def x(self) -> List[Path]:
        if self._lazy_x is None:
            x = np.array([])
            if self.train:
                files = list(self.files_in_folder)
                number_files = self.number_files_in_folders
                min_n_files = np.min(number_files)
                for index in range(min_n_files):
                    for index_label, n_files in enumerate(number_files):
                        index_file = index % n_files
                        file = files[index_label][index_file]
                        x = np.append(x,file)
            else:
                x = sum(list(self.files_in_folder), [])
            self._lazy_x = x
        return self._lazy_x

    def partition(
        self,
        val_size: float = 0.2,
        tamanho: Union[int,None] = None,
        shuffle: bool = True
    ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
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
        if tamanho is None or tamanho > len(self.x) or tamanho < 1:
            tamanho = len(self.x)
        x = self.x[:tamanho]
        y = self.y[:tamanho]
        train_in, val_in, train_out, val_out = train_test_split(
            x, y,
            test_size=val_size,
            shuffle=shuffle
        )
        train, val = (train_in, train_out), (val_in, val_out)
        return train, val

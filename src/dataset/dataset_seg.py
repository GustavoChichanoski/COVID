"""
    Gera o dataset.
"""
from typing import Any, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SegmentationDataset:
    """ Cria o dataset para o keras. """
    path_lung: Path
    path_mask: Path = None
    """
        Args:
            path_data (str): Caminho onde se encontra os dados dos raios-x
            number_splits (int): numero de cortes por imagem.
            dimension_original (int): dimensão da imagem original
            dimension_cut (int): dimensão dos recortes
    """
    _lazy_x: Optional[List[Path]] = None
    _lazy_y: Optional[Any] = None

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
    def number_files_in_folders(self) -> List[List[int]]:
        if self._lazy_number_files_in_folders is None:
            files = np.array([ len(folder) for folder in self.files_in_folder ])
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
        """
            Generate the y values of inputs images based in your class

            Returns:
                numpy.array: the classes of images
        """
        if self._lazy_y is None:
            y = np.array([])
            for lung_id in self.x.iterdir():
                if lung_id.find('CHN'):
                    mask_id = self.change_extension(lung_id,'_mask.png')
                    path_mask = self.path_mask / mask_id
                    if not path_mask.exists():
                        raise ValueError(f'O caminho {path_mask} não é valido')
                else:
                    path_mask = self.path_mask / lung_id
                y = np.append(y,path_mask)
            self._lazy_y = y
        return self._lazy_y

    def change_extension(
        self,
        path: Path,
        new_extension: str
    ) -> Path:
        filename = str(path).split(path.suffix)[0]
        return filename + new_extension

    @property
    def x(self) -> List[Path]:
        if self._lazy_x is None:
            x = sum(list(self.files_in_folder), [])
            self._lazy_x = x
        return self._lazy_x

    def partition(
        self,
        val_size: float = 0.2,
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
        tam_max = tamanho if tamanho > 0 and tamanho < len(self.x) else len(self.x)
        x, y = self.x[:tam_max], self.y[:tam_max]
        train_in, val_in, train_out, val_out = train_test_split(
            x,
            y,
            test_size=val_size,
            shuffle=shuffle
        )
        train, val = (train_in, train_out), (val_in, val_out)
        return train, val

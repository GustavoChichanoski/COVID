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
    path_mask: Optional[Path]= None
    """
        Args:
            path_data (str): Caminho onde se encontra os dados dos raios-x
            number_splits (int): numero de cortes por imagem.
            dimension_original (int): dimensão da imagem original
            dimension_cut (int): dimensão dos recortes
    """
    _lazy_x: Optional[List[Path]] = None
    _lazy_y: Optional[List[Path]] = None

    @property
    def y(self) -> List[Path]:
        """
            Generate the y values of inputs images based in your class

            Returns:
                numpy.array: the classes of images
        """
        if self._lazy_y is None:
            if self.path_mask is not None:
                y = np.array([])
                y = [path for path in self.path_mask.iterdir()]
                self._lazy_y = y
            else:
                self._lazy_y = self.x
        return self._lazy_y

    def change_extension(self, filename: str, old: str, new: str) -> str:
        file_id = filename.split(old)[0]
        file_id = f'{file_id}{new}'
        return file_id

    @property
    def x(self) -> List[Path]:
        if self._lazy_x is None:
            x = []
            if self.path_mask is None:
                x = [lung_id for lung_id in self.path_lung.iterdir()]
            else:
                for mask_id in self.y:
                    filename = mask_id.parts[-1]
                    if filename.startswith('CHN'):
                        lung_id = filename.replace('_mask.png', '.png')
                        lung = self.path_lung / lung_id
                    else:
                        lung = self.path_lung / filename
                    if lung.exists():
                        x.append(lung)
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
        x = self.x[:tam_max]
        y = self.y[:tam_max]

        x = np.array(x)
        y = np.array(y)
        train_in, val_in, train_out, val_out = train_test_split(
            x, y, test_size=val_size, shuffle=shuffle
        )
        train, val = (train_in, train_out), (val_in, val_out)
        return train, val

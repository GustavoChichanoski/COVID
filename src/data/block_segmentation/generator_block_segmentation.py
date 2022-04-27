from src.images.read_image import read_step
import tensorflow_addons as tfa
import pandas as pd
from tensorflow.python.keras.utils.all_utils import Sequence
from typing import List, Optional, Tuple, Union
import numpy as np

class GeneratorBlockSegmentation(Sequence):

    def __init__(
        self,
        df: pd.DataFrame,
        x_col_name: str,
        y_col_name: Optional[Union[str, List[str]]] = None,
        mask_col_name: Optional[str] = None,
        batch_size: int = 2,
        dim: int = 256,
        channels: int = 1
    ) -> None:
        super(GeneratorBlockSegmentation).__init__()
        self._lazy_x = None
        self._lazy_y = None
        self.df = df
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.mask_col_name = mask_col_name
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels

    @property
    def x(self) -> tfa.types.TensorLike:
        if self._lazy_x is None:
            self._lazy_x = self.df[self.x_col_name].values
        return self._lazy_x
    
    @property
    def y(self) -> tfa.types.TensorLike:
        if self._lazy_y is None:
            if self.mask_col_name is not None:
                self._lazy_y = self.df[self.mask_col_name].values
            else:
                col = np.array([])
                classes = len(self.y_col_name)
                for col_name in self.y_col_name:
                    col = np.append(col,self.df[col_name].values)
                col = np.reshape(col,(int(col.size/classes),classes))
                self._lazy_y = col
        return self._lazy_y

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        length = int(np.floor(len(self.x) / self.batch_size))
        return length

    def __getitem__(self, index: int) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:
        idi = index * self.batch_size
        idf = (index + 1) * self.batch_size
        x = self.x[idi:idf]
        y = self.y[idi:idf]
        shape = (self.batch_size,self.dim,self.dim,self.channels)
        x = read_step(x,shape)
        if self.mask_col_name is not None:
            y = read_step(y,shape)
        return (x,y)
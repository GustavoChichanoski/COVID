from pathlib import Path
from typing import Any, List, Tuple
from tensorflow.python.keras.utils.all_utils import Sequence
from src.images.read_image import read_images
from src.images.process_images import split, split_images_n_times as split_images
from src.images.read_image import read_images
import numpy as np

class DataGenerator(Sequence):

    def __init__(
        self,
        x_set,
        y_set,
        batch_size: int = 64,
        dim: int = 224,
        n_class: int = 3,
        channels: int = 3,
        threshold: float = 0.45
    ) -> None:
        """[Initialize the Datagenerator]

        Args:
            x_set (np.array(Path)): list of paths content images paths
            y_set (np.array(float)): outputs values of images
            batch_size (int, optional):
                batch size per get. Defaults to 64.
            dim (int, optional):
                dimension of splits. Defaults to 224.
            n_class (int, optional):
                number of class of your model. Defaults to 3.
            channels (int, optional):
                number of channel of images. Defaults to 3.
            threshold (float, optional):
                the minimum precent valid pixel in split image.
                Defaults to 0.45.
        """    
        self.x, self.y = x_set, y_set
        self.batch_size = len(self.x) if batch_size > len(self.x) else batch_size
        self.dim = dim
        self.n_class = n_class
        self.channels = channels
        self.threshold = threshold

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[Any,Any]:
        """
            Get th data of dataset with position initial in idx to idx plus batch_size.

            Args:
                idx (int): initial position

            Returns:
                (Any,Any): the first term is x values of dataset
                           the second term is y vlaues of dataset
        """
        idi = idx * self.batch_size
        idf = (idx + 1) * self.batch_size
        batch_x, batch_y = self.x[idi:idf], self.y[idi:idf]
        y_shape = (self.batch_size, self.n_class)
        batch_y = batch_y.reshape(y_shape)
        params_splits = {
            'verbose': False,
            'dim': self.dim,
            'channels': self.channels,
            'threshold': self.threshold,
            'n_splits': 1
        }
        batch_x, _positions = split(batch_x, **params_splits)
        return batch_x, batch_y
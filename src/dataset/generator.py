from pathlib import Path
from typing import Any, List
from tensorflow.python.keras.utils.all_utils import Sequence
from src.images.read_image import read_images
from src.images.process_images import split_images_n_times as split_images
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
        self.x, self.y = x_set, y_set
        self.batch_size = len(self.x) if batch_size < len(self.x) else batch_size
        self.dim = dim
        self.n_class = n_class
        self.channels = channels
        self.threshold = threshold
        self._lazy_id_inicial = None

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        shape = (self.batch_size, self.n_class)
        batch_y = batch_y.reshape(shape)
        batch_x = self.split(batch_x)
        return batch_x, batch_y

    def split(
        self,
        paths_images_in_batch: List[Path],
    ) -> Any:
        shape = (self.batch_size, self.dim, self.dim, self.channels)
        images = (read_images(path) for path in paths_images_in_batch)
        params = {'n_split': 1, 'dim_split': self.dim, 'verbose': False,
                  'need_positions': False, 'threshold': self.threshold}
        splited_images = [split_images(image, **params) for image in images]
        cuts = np.array(splited_images)
        cuts_reshapeds = cuts.reshape(shape)
        return cuts_reshapeds
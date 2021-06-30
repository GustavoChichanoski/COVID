from abc import abstractmethod
from typing import Any, Tuple
from tensorflow.python.keras.utils.all_utils import Sequence
from src.images.process_images import split
import numpy as np
import tensorflow_addons as tfa

class KerasGenerator(Sequence):

    def __init__(
        self,
        x_set: tfa.types.TensorLike,
        y_set: tfa.types.TensorLike,
        batch_size: int = 64,
        dim: int = 224,
        n_class: int = 1,
        channels: int = 1,
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

    def __getitem__(self, index: int) -> tfa.types.TensorLike:
        angle = np.random.rand(self.batch_size)
        idi = index * self.batch_size
        idf = (index + 1) * self.batch_size
        batch_x = self.x[idi:idf]
        batch_x = self.step(batch_x, angle)
        if self.y is not None:
            batch_y = self.y[idi:idf]
            batch_y = self.step(batch_y, angle)
            return batch_x, batch_y
        return batch_x

    @abstractmethod
    def step(self, angle: float = 0.0) -> tfa.types.TensorLike:
        raise NotImplementedError("Must override with correct step read")
from abc import abstractmethod
from typing import Any, Tuple
from tensorflow.python.keras.utils.all_utils import Sequence
from src.images.process_images import split
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import math

class KerasGenerator(Sequence):

    def __init__(
        self,
        x_set: tfa.types.TensorLike,
        y_set: tfa.types.TensorLike,
        batch_size: int = 64,
        dim: int = 224,
        n_class: int = 1,
        channels: int = 1,
        threshold: float = 0.45,
        angle: float = 5.0,
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        rotate_image: bool = True,
        filter_mean: bool = True
    ) -> None:
        """ Based class to generate dataset of keras

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
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.rotate_image = rotate_image
        self.filter_mean = filter_mean
        self.angle = angle

    def generate_random_angle(self):
        max_angle = self.angle * math.pi / 180
        rotation = tf.random.uniform([],-max_angle,max_angle,dtype=tf.float32)
        return rotation

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index: int) -> tfa.types.TensorLike:
        angle = self.generate_random_angle()
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
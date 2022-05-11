from abc import abstractmethod
from typing import Any, Tuple
from tensorflow.python.keras.utils.data_utils import Sequence
from src.images.process_images import split
import albumentations as A
from albumentations import Compose
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
        compose: Compose = None,
        desire_len: int = None
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
        self.len_x = len(self.x)
        self.batch_size = batch_size
        if batch_size > len(self.x):
            self.batch_size = len(self.x)
        self.dim = dim
        self.n_class = n_class
        self.channels = channels
        self.threshold = threshold
        self.desire_len = desire_len
        if compose is None:
            self.compose = compose
        else:
            self.compose = Compose([
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.JpegCompression(quality_lower=80, quality_upper=90),
                A.Rotate(),
                A.OpticalDistortion(),
            ])

    def generate_random_angle(self):
        max_angle = self.angle * math.pi / 180
        rotation = tf.random.uniform(
            [], -max_angle, max_angle, dtype=tf.float32)
        return rotation

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        if self.desire_len is None:
            return int(np.floor(self.len_x / self.batch_size))
        return int(np.floor(self.desire_len / self.batch_size))

    def __getitem__(self, index: int) -> tfa.types.TensorLike:
        idi = index * self.batch_size
        idf = (index + 1) * self.batch_size

        idi = idi % self.len_x
        idf = idf % self.len_x

        batch_x, batch_y = self.completar_array(idi, idf)
        batch_x = self.step_x(batch_x)
        if self.y is not None:
            batch_y = self.y[idi:idf]
            batch_y = self.step_y(batch_y)
            return batch_x, batch_y
        return batch_x

    def completar_array(self,
                        indice_inicial: int = 0,
                        indice_final: int = 0):
        if indice_final > self.len_x:
            indice_inicial = int(indice_inicial % self.len_x)
            indice_final = int(indice_final % self.len_x)
            batch_x = self.x[indice_inicial:self.len_x]
            batch_y = self.y[indice_inicial:self.len_x]
            for indice in range(indice_inicial - indice_final):
                image = self.compose(self.x[indice])
                batch_x = np.append(batch_x, image)
                batch_y = np.append(batch_y, self.y[indice])
            return batch_x, batch_y
        batch_x = self.x[indice_inicial:indice_final]
        batch_y = self.y[indice_inicial:indice_final]
        return batch_x, batch_y

    @abstractmethod
    def step_x(self) -> tfa.types.TensorLike:
        raise NotImplementedError("Must override with correct step read")

    @abstractmethod
    def step_y(self) -> tfa.types.TensorLike:
        raise NotImplementedError("Must override with correct step read")

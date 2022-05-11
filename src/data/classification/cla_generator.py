from typing import Any, Tuple

import numpy as np
from src.images.process_images import split
from src.data.generator import KerasGenerator
import tensorflow_addons as tfa

class ClassificationDatasetGenerator(KerasGenerator):

    def __init__(
        self,
        x_set: tfa.types.TensorLike,
        y_set: tfa.types.TensorLike,
        threshold: float = 0.45,
        **params
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
        super().__init__(x_set=x_set,y_set=y_set,**params)
        self.threshold = threshold

    def step_x(
        self, batch: tfa.types.TensorLike, angle: float = 0
    ) -> tfa.types.TensorLike:
        """
            Get th data of dataset with position initial in idx to idx plus batch_size.

            Args:
                idx (int): initial position

            Returns:
                (Any,Any): the first term is x values of dataset
                           the second term is y vlaues of dataset
        """
        params_splits = {
            'verbose': False,
            'dim': self.dim,
            'channels': self.channels,
            'threshold': self.threshold,
            'n_splits': 1
        }
        batch_x, _positions = split(batch, **params_splits)
        new_shape = (self.batch_size, self.dim, self.dim, self.channels)
        batch_x = np.reshape(batch_x,new_shape)
        return batch_x

    def step_y(self, batch: tfa.types.TensorLike, angle: float = 0.0) -> tfa.types.TensorLike:
        """
            Get th data of dataset with position initial in idx to idx plus batch_size.

            Args:
                idx (int): initial position

            Returns:
                (Any,Any): the first term is x values of dataset
                           the second term is y vlaues of dataset
        """
        y_shape = (self.batch_size, self.n_class)
        batch_y = batch.reshape(y_shape)
        return batch_y

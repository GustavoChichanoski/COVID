from typing import Any, Optional, Tuple
from src.images.read_image import read_step
from src.images.process_images import augmentation_image
from src.dataset.generator import KerasGenerator
from src.dataset.dataset import Dataset
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

class SegmentationDatasetGenerator(KerasGenerator):

    def __init__(
        self,
        x_set: tfa.types.TensorLike,
        y_set: Optional[tfa.types.TensorLike],
        augmentation: bool = False,
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
        sharpness: bool = False,
        mean_filter: bool = False,
        angle: float = 0.1,
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
        """
        super().__init__(
            x_set=x_set,
            y_set=y_set,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            angle=angle,
            **params
        )
        self.mean_filter = mean_filter

    def step(
        self,
        batch: tfa.types.TensorLike,
        angle: float = 0
    ) -> Tuple[Any,Any]:
        """
            Get th data of dataset with position initial in idx to idx plus batch_size.

            Args:
                idx (int): initial position

            Returns:
                (Any,Any): the first term is x values of dataset
                           the second term is y vlaues of dataset
        """
        shape = (len(batch), self.dim,self.dim,self.channels)
        batch = read_step(batch, shape)
        batch = (batch / 255.0).astype(np.float32)
        if self.augmentation:
            batch = augmentation_image(
                batch, angle, self.flip_horizontal,
                self.flip_vertical, self.mean_filter
            )

        shape = (int(batch.size / (self.dim * self.dim)),self.dim,self.dim,self.channels)
        batch = batch.reshape(shape)
        return batch
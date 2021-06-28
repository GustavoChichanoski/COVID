from typing import Any, Tuple
from tensorflow.python.keras.utils.all_utils import Sequence
from src.images.read_image import read_step
from src.images.process_images import augmentation_image
import tensorflow as tf
import numpy as np

class SegmentationDataGenerator(Sequence):

    def __init__(
        self,
        x_set: Any,
        y_set: Any,
        augmentation: bool = False,
        batch_size: int = 64,
        dim: int = 224
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
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.dim = dim
        self.augmentation = augmentation

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
        batch_x = self.x[idi:idf]
        batch_y = self.y[idi:idf]

        shape = (len(batch_x),self.dim,self.dim,1)
        batch_x = read_step(batch_x, shape)
        batch_y = read_step(batch_y, shape)

        batch_x = (batch_x / 255.0).astype(np.float32)
        batch_y = (batch_y > 127).astype(np.float32)

        if self.augmentation:
            batch_x, batch_y = augmentation_image(batch_x, batch_y)
        total = 1
        for shape in list(batch_x.shape):
            total *= shape

        shape = (int(total / (self.dim * self.dim)),self.dim,self.dim,1)

        batch_x = batch_x.reshape(shape)
        batch_y = batch_y.reshape(shape)

        return batch_x, batch_y
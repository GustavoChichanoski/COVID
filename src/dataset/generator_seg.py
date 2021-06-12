from pathlib import Path
from typing import Any, List, Tuple
from tensorflow.python.keras.utils.all_utils import Sequence
from src.images.read_image import read_images
from src.images.process_images import split, split_images_n_times as split_images
from src.images.read_image import read_images
import numpy as np

class SegmentationDataGenerator(Sequence):

    def __init__(
        self,
        x_set,
        y_set,
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
        shape = (self.batch_size, self.dim, self.dim, 1)

        idi = idx * self.batch_size
        idf = (idx + 1) * self.batch_size
        batch_x, batch_y = self.x[idi:idf], self.y[idi:idf]

        batch_x = self.read_step(batch_x)
        batch_y = self.read_step(batch_y)

        batch_x = np.reshape(batch_x, shape)
        batch_y = np.reshape(batch_y, shape)


        batch_x = (batch_x / 255).astype(np.float32)
        batch_y = (batch_y > 127).astype(np.float32)

        return batch_x, batch_y

    def read_step(self, images: Any) -> Any:
        read_params = { 'color': False, 'output_dim': self.dim }
        batch = [read_images(image, **read_params) for image in images]
        return np.array(batch)
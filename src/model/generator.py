from src.images.read_image import read_images
from typing import List
import numpy as np
from tensorflow.python.keras.utils.all_utils import Sequence
from src.dataset.dataset import Dataset
from src.images.process_images import split_images
from src.images.read_image import read_images

class DataGenerator(Sequence):

    def __init__(self,
                 data,
                 batch_size: int = 64,
                 dim: int = 224,
                 shuffle: bool = True,
                 n_class: int = 3,
                 channels: int = 3):
        self.x, self.y = data
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.n_class = n_class
        self.channels = channels

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:
                         (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:
                         (idx + 1) * self.batch_size]
        batch_y.shape = (self.batch_size, self.n_class)
        batch_x = self.split(batch_x, self.batch_size)
        return batch_x, batch_y

    def split(self, paths_images_in_batch, batch_size):
        images = (read_images(path) for path in paths_images_in_batch)
        splited_images = [split_images(image,self.dim) for image in images]
        cuts = np.array(splited_images)
        cuts.shape = (batch_size, self.dim, self.dim, self.channels)
        return cuts

from typing import List
import numpy as np
import math
from keras.utils import Sequence
from src.dataset.dataset import Dataset
from src.images.process_images import split_images


class DataGenerator(Sequence):

    def __init__(self,
                 data,
                 labels: List[str],
                 batch_size: int = 64,
                 dim: int = 224,
                 shuffle: bool = True,
                 n_class: int = 3,
                 channels: int = 3,
                 train: bool = True):
        self.train = train
        self.x, self.y = data
        self.batch_size = batch_size
        self.dim = dim
        self.ids = 0
        self.labels = labels
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
        batch_y = np.array(batch_y)
        batch_y = batch_y.reshape(self.batch_size, self.n_class)
        batch_x = self.split(batch_x, self.batch_size)
        return batch_x, batch_y

    def split(self, paths, batch_size):
        dim_split = self.dim
        channels = self.channels
        cuts = []
        for path in paths:
            cuts = np.append(cuts, split_images(path,dim_split))
        cuts = cuts.reshape(batch_size, dim_split, dim_split, channels)
        return cuts

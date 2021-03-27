from keras import Model
from src.images.process_images import split_images
import tensorflow as tf
import numpy as np


class CustomFit(Model):

    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model

    def compile(self,optimizer, loss):
        super(CustomFit, self).compiele()
        self.optimizer = optimizer
        self.loss = loss


    def train_step(self, data):
        # x é os caminhos de todas as imagens
        # x é do tamanho do batch size
        # y é a saída de todas as imagens
        if isinstance(data, tuple):
            data = data[0]
        batch_size = tf.shape(data)[0]
        

    def split(self,path, batch_size):
        dim_split = self.input_shape[1]
        channels = self.input_shape[-1]
        cuts = []
        for path in paths:
            cuts = np.append(cuts, split_images(path))
        cuts = cuts.reshape(batch_size,dim_split,dim_split,channels)
        return cuts
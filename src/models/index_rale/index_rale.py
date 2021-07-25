import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd

class Rale(tf.python.keras.Model):

    def __init__(
        self,
        x: tfa.types.TensorLike,
        y: tfa.types.TensorLike
    ) -> None:
        super().__init__()
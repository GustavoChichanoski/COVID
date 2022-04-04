import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import PreprocessingLayer

class CutBlock(PreprocessingLayer):

  def __init__(
    self,
    prob: float = 0.5,
    streaming: bool = True,
    **kwargs
  ) -> None:
      super().__init__(streaming, **kwargs)
      self.prob = prob

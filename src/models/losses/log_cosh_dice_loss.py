from src.models.losses.dice_loss import DiceError
from typing import Any
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa

class LogCoshDiceError(DiceError):
    
    def __init__(
        self,
        regularization_factor: float = 1,
        name: str = 'log_cosh_f1'
    ) -> None:
        super().__init__(
            name=name,
            regularization_factor=regularization_factor
        )
        
    def call(
        self,
        y_true: tfa.types.TensorLike,
        y_pred: tfa.types.TensorLike
    ) -> tfa.types.TensorLike:
        dice_loss = super().call(y_true=y_true,y_pred=y_pred)
        total_loss = tf.math.log(tf.math.cosh(dice_loss))
        total_loss = total_loss * self.regularization_factor
        return total_loss
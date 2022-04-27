from typing import Any, Optional
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa

class DiceError(Loss):
    
    def __init__(
        self,
        regularization_factor: Optional[float] = 1,
        name: str = 'f1'
    ) -> None:
        if regularization_factor is not None:
            self.regularization_factor = regularization_factor
        else:
            self.regularization_factor = losses_utils.ReductionV2.AUTO
        super().__init__(name=name)

    def call(
        self,
        y_true: tfa.types.TensorLike,
        y_pred: tfa.types.TensorLike
    ) -> tfa.types.TensorLike:

        class_num = y_true.shape[-1] if y_true.shape[-1] is not None else 1
        total_loss = 0

        for class_now in range(class_num):

            len_shape = len(y_true.shape)

            if len_shape == 4 and class_num > 1:
                # Converte y_pred e y_true em vetores
                y_true = y_true[:,:,:,class_now]
                y_pred = y_pred[:,:,:,class_now]
            elif len_shape == 3 and class_num > 1:
                # Converte y_pred e y_true em vetores
                y_true = y_true[:,:,class_now]
                y_pred = y_pred[:,:,class_now]
            elif len_shape == 2 and class_num > 1:
                # Converte y_pred e y_true em vetores
                y_true = y_true[:,class_now]
                y_pred = y_pred[:,class_now]
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)

            loss = dice_loss(y_true_f, y_pred_f)

            total_loss = total_loss + loss

        total_loss = total_loss / class_num
        total_loss = (1 - total_loss) * self.regularization_factor
        return total_loss

def dice_loss(
    y_true_f: tfa.types.TensorLike,
    y_pred_f: tfa.types.TensorLike
) -> tfa.types.TensorLike:
    y_true_f = tf.cast(y_true_f, dtype=tf.float64)
    y_pred_f = tf.cast(y_pred_f, dtype=tf.float64)
    # Correção da equação de erro
    correction_factor = tf.constant(1.0, dtype=tf.float64)
    # Smooth - Evita que o denominador fique muito pequeno
    smooth = K.constant(1e-6, dtype=tf.float64)
    # Calculo o erro entre eles
    constant = K.constant(2.0, dtype=tf.float64)

    # Calcula o numero de vezes que
    # y_true(positve) é igual y_pred(positive) (tp)
    intersection = K.sum(y_true_f * y_pred_f)
    # Soma o número de vezes que ambos foram positivos
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    num = tf.math.multiply(constant,intersection) + correction_factor
    den = union + smooth + correction_factor
    loss = num / den
    return loss
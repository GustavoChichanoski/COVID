from typing import Any
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras import backend as K
import tensorflow as tf

class LogCoshDiceError(Loss):
    
    def __init__(
        self,
        regularization_factor: float = 1,
        name: str = 'log_cosh_f1'
    ) -> None:
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true: Any, y_pred: Any):

        class_num = y_true.shape[-1] if y_true.shape[-1] is not None else 1
        total_loss = 0

        for class_now in range(class_num):

            # Converte y_pred e y_true em vetores
            y_true_f = K.flatten(y_true[:,:,:,class_now])
            y_pred_f = K.flatten(y_pred[:,:,:,class_now])

            one = tf.constant(1.0, dtype=tf.float32)

            # Calcula o numero de vezes que
            # y_true(positve) é igual y_pred(positive) (tp)
            intersection = K.sum(y_true_f * y_pred_f)
            # Soma o número de vezes que ambos foram positivos
            union = K.sum(y_true_f) + K.sum(y_pred_f)
            # Smooth - Evita que o denominador fique muito pequeno
            smooth = K.constant(1e-6, dtype=tf.float32)
            # Calculo o erro entre eles
            constant = K.constant(2.0, dtype=tf.float32)
            num = constant * intersection + one
            den = union + smooth + one
            loss = num / den
            
            if class_now == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        
            total_loss = total_loss / class_num

        return tf.math.log(tf.math.cosh(1 - total_loss)) * self.regularization_factor
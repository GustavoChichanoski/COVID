from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa

from src.models.losses.dice_loss import dice_loss

class UNetPlusPlus(Loss):
    
    def __init__(
        self,
        from_logits: bool = False,
        regularization: float = 1,
        name: str = 'unet++'
    ) -> None:
        super().__init__(
            name=name,
            from_logits=from_logits
        )
        self.regularization = regularization
        
    def call(
        self,
        y_true: tfa.types.TensorLike,
        y_pred: tfa.types.TensorLike
    ) -> tfa.types.TensorLike:
        
        class_num = y_true.shape[-1] if y_true.shape[-1] is not None else 1
        batch_size = y_true.shape[0]
        total_loss = 0

        for class_now in range(class_num):

            for batch in range(batch_size):
                # Converte y_pred e y_true em vetores
                y_true_f = K.flatten(y_true[batch,:,:,class_now])
                y_pred_f = K.flatten(y_pred[batch,:,:,class_now])
                batch_loss = tf.multiply(y_true_f,tf.math.log(y_pred_f))
                dice = tf.multiply(2,tf.multiply(y_true_f,y_pred_f))
                dice = tf.divide(dice,tf.sum(y_true_f,y_pred_f))
                batch_loss += dice

                total_loss += batch_loss

            total_loss = -total_loss/batch_size
            
        total_loss = total_loss / class_num

        return total_loss * self.regularization
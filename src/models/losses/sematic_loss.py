from typing import Any
from numpy import intersect1d
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from src.models.losses.log_cosh_dice_loss import LogCoshDiceError


class SemanticLossFunctions(LogCoshDiceError):

  def __init__(
    self,
    regularization_factor: float = 1,
    name: str = 'log_cosh_f1',
    smooth: float = 1.0,
    beta: float = 0.25,
    alpha: float = 0.25,
    gamma: float = 2
  ) -> None:
    super().__init__(regularization_factor, name)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.smooth = smooth

  def dice_coef(
      self, y_true: tfa.types.TensorLike, y_pred: tfa.types.TensorLike
  ) -> tfa.types.TensorLike:
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    num = (2 * intersection + K.epsilon())
    den = (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return num / den

  def sensitivity(
      self, y_true: tfa.types.TensorLike, y_pred: tfa.types.TensorLike
  ) -> tfa.types.TensorLike:
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (true_positives + K.epsilon())

  def specifity(
      self, y_true: tfa.types.TensorLike, y_pred: tfa.types.TensorLike
  ) -> tfa.types.TensorLike:
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

  def covert_to_logits(self, y_pred: tfa.types.TensorLike) -> tfa.types.TensorLike:
    y_pred = tf.clip_by_value(
      y_pred,
      tf.keras.backend.epsilon(),
      1 - tf.keras.backend.epsilon()
    )
    return tf.math.log(y_pred / (1 - y_pred))

  def weighted_cross_entropyloss(
    self, y_true: tfa.types.TensorLike, y_pred: tfa.types.TensorLike
  ) -> tfa.types.TensorLike:
    y_pred = self.convert_to_logits(y_pred)
    pos_weight = self.beta / (1 - self.beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight)
    return tf.reduce_mean(loss)

  def focal_loss_with_logits(
    self,
    logits: tfa.types.TensorLike,
    targets: tfa.types.TensorLike,
    alpha: tfa.types.TensorLike,
    gamma: tfa.types.TensorLike,
    y_pred: tfa.types.TensorLike
  ) -> tfa.types.TensorLike:
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    calc_relu = tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)
    calc_weight_b = logits * weight_b
    return calc_relu * (weight_a + weight_b) + calc_weight_b

  def focal_loss(
    self,
    y_true: tfa.types.TensorLike,
    y_pred: tfa.types.TensorLike
  ) -> tfa.types.TensorLike:
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                       alpha=self.alpha, gamma=self.gamma,
                                       y_pred=y_pred)
    return tf.reduce_mean(loss)

  def depth_softmax(self, matrix: tfa.types.TensorLike) -> tfa.types.TensorLike:
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

  def dice_loss(
    self,
    y_true: tfa.types.TensorLike,
    y_pred: tfa.types.TensorLike
  ) -> tfa.types.TensorLike:
    loss = self.super().call(y_true=y_true, y_pred=y_pred)
    return

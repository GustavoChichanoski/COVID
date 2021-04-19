import tensorflow as tf
import keras
from keras.metrics import Metric
import keras.backend as K

class F1score(Metric):

    def __init__(self, name='f1', **kwargs) -> None:
        super(F1score, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='tp', initializer='zeros')

    def result(self):
        return self.score

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(K.round(K.clip(y_true, 0, 1)), tf.bool)
        y_pred = tf.cast(K.round(K.clip(y_pred, 0, 1)), tf.bool)
        tp = self.true_positive(y_true, y_pred, sample_weight)
        tn = self.true_negative(y_true, y_pred, sample_weight)
        fp = self.false_positive(y_true, y_pred, sample_weight)
        fn = self.false_negative(y_true, y_pred, sample_weight)
        self.score = self.score.assign_add(
            tf.reduce_sum(self.f1_score(tp, tn, fp, fn))
        )


    def false_negative(self, y_true, y_pred, sample_weight):
        tn = tf.logical_and(tf.equal(y_true, True),
                            tf.equal(y_pred, False))
        tn = tf.cast(tn, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(tn, self.dtype)
            tn = tf.multiply(tn, sample_weight)
        return tf.reduce_sum(tn)

    def false_positive(self, y_true, y_pred, sample_weight):
        tn = tf.logical_and(tf.equal(y_true, False),
                            tf.equal(y_pred, True))
        tn = tf.cast(tn, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(tn, self.dtype)
            tn = tf.multiply(tn, sample_weight)
        return tf.reduce_sum(tn)

    def true_negative(self, y_true, y_pred, sample_weight):
        tn = tf.logical_and(tf.equal(y_true, False),
                            tf.equal(y_pred, False))
        tn = tf.cast(tn, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(tn, self.dtype)
            tn = tf.multiply(tn, sample_weight)
        return tf.reduce_sum(tn)

    def true_positive(self, y_true, y_pred, sample_weight):
        tp = tf.logical_and(tf.equal(y_true, True),
                            tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(tp, self.dtype)
            tp = tf.multiply(tp, sample_weight)
        return tf.reduce_sum(tp)

    def f1_score(self, tp, tn, fp, fn):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall)

    def reset_state(self):
        self.score.assign(0)
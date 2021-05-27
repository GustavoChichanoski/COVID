import tensorflow as tf
from tensorflow.python.keras import metrics
from tensorflow.python.keras.metrics import Metric
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
import numpy as np

class F1score(Metric):
    def __init__(
        self,
        name: str = "f1",
        dtype=None,
        thresholds: float = None,
        top_k=None,
        class_id=None,
        **kwargs
    ) -> None:
        super(F1score, self).__init__(name=name, dtype=dtype, **kwargs)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds=thresholds, default_threshold=default_threshold
        )
        self.true_positives = self.add_weight(
            "true_positives",
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer,
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer,
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer,
        )
        self.score = self.add_weight(name="f1", initializer="zeros")

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        precision = precision[0] if len(self.thresholds) == 1 else precision
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        recall = recall[0] if len(self.thresholds) == 1 else recall
        score = tf.math.multiply_no_nan(precision, recall)
        score = tf.math.multiply_no_nan(tf.constant(2.0), score)
        return tf.math.divide_no_nan(score, precision + recall)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true=y_true,
            y_pred=y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in (
                    self.true_positives,
                    self.false_positives,
                    self.false_negatives,
                )
            ]
        )

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super(F1score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    accuracy = np.mean(y_true == y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    class_metrics = {}
    for i, name in enumerate(class_names):
        if i < len(precision):
            class_metrics[name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'class_metrics': class_metrics,
        'macro_avg': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1)
        },
        'weighted_avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1_score': float(weighted_f1)
        }
    }


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> str:
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

    return float(np.mean(correct))

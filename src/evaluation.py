from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_metrics(y_true, y_pred) -> dict:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }


"""
Confusion-matrix based metrics for change detection.
"""

from __future__ import annotations

import numpy as np


class AverageMeter:
    """Computes and stores running average and current value."""

    def __init__(self) -> None:
        self.initialized = False
        self.val = self.avg = self.sum = self.count = None

    def initialize(self, val, weight) -> None:
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight: float = 1) -> None:
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.val = val
            self.sum += val * weight
            self.count += weight
            self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self) -> dict:
        return cm2score(self.sum)

    def clear(self) -> None:
        self.initialized = False


class ConfuseMatrixMeter(AverageMeter):
    """Confusion-matrix tracker supporting F1, IoU and OA metrics."""

    def __init__(self, n_class: int) -> None:
        super().__init__()
        self.n_class = n_class

    def update_cm(self, pr: np.ndarray, gt: np.ndarray, weight: float = 1) -> float:
        """Update confusion matrix and return current mean F1."""
        val = get_confuse_matrix(self.n_class, gt, pr)
        self.update(val, weight)
        return float(cm2F1(val))

    def get_scores(self) -> dict:
        return cm2score(self.sum)


# ---------------------------------------------------------------------------
# Score computation helpers
# ---------------------------------------------------------------------------


def cm2F1(cm: np.ndarray) -> float:
    """Mean F1 score from confusion matrix *cm*."""
    tp = np.diag(cm)
    precision = tp / (cm.sum(axis=0) + np.finfo(np.float32).eps)
    recall = tp / (cm.sum(axis=1) + np.finfo(np.float32).eps)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return float(np.nanmean(F1))


def cm2score(cm: np.ndarray) -> dict:
    """Compute accuracy, IoU, F1 and per-class metrics from *cm*."""
    eps = np.finfo(np.float32).eps
    tp = np.diag(cm)
    sum_rows = cm.sum(axis=1)
    sum_cols = cm.sum(axis=0)

    acc = tp.sum() / (cm.sum() + eps)
    recall = tp / (sum_rows + eps)
    precision = tp / (sum_cols + eps)
    F1 = 2 * recall * precision / (recall + precision + eps)
    iu = tp / (sum_rows + sum_cols - tp + eps)

    n = cm.shape[0]
    scores = {
        "acc": float(acc),
        "miou": float(np.nanmean(iu)),
        "mf1": float(np.nanmean(F1)),
    }
    scores.update({f"iou_{i}": float(iu[i]) for i in range(n)})
    scores.update({f"F1_{i}": float(F1[i]) for i in range(n)})
    scores.update({f"precision_{i}": float(precision[i]) for i in range(n)})
    scores.update({f"recall_{i}": float(recall[i]) for i in range(n)})
    return scores


def get_confuse_matrix(
    num_classes: int,
    label_gts: np.ndarray,
    label_preds: np.ndarray,
) -> np.ndarray:
    """Build a confusion matrix from arrays of ground truth and prediction."""

    def _fast_hist(gt, pred):
        mask = (gt >= 0) & (gt < num_classes)
        return np.bincount(
            num_classes * gt[mask].astype(int) + pred[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)

    cm = np.zeros((num_classes, num_classes))
    for gt, pred in zip(label_gts, label_preds):
        cm += _fast_hist(gt.flatten(), pred.flatten())
    return cm

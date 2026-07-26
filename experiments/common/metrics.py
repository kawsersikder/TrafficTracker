"""Evaluation metrics.

Headline metric for the paper: congestion-onset F1 — did the model predict the
jump into MH-or-worse (CI >= 0.6) at timesteps where the previous ground-truth
state was below the threshold?  This is what matters to practitioners and is
where class imbalance (H = 0.8% of Manila rows) actually bites.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from .data import CI_CLASS_BINS, ONSET_CLS_THRESHOLD, ci_to_cls


def onset_f1(true_cls: np.ndarray, pred_cls: np.ndarray, prev_true_cls: np.ndarray,
             thr: int = ONSET_CLS_THRESHOLD) -> dict:
    eligible = prev_true_cls < thr          # onset only defined from a calm state
    t = (true_cls >= thr) & eligible
    p = (pred_cls >= thr) & eligible
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"onset_precision": prec, "onset_recall": rec, "onset_f1": f1,
            "onset_events": int(t.sum())}


def classification_report(true_cls, pred_cls, prev_true_cls) -> dict:
    rep = {
        "accuracy": float((true_cls == pred_cls).mean()),
        "macro_f1": float(f1_score(true_cls, pred_cls, average="macro", zero_division=0)),
    }
    for c, f1c in enumerate(f1_score(true_cls, pred_cls, average=None,
                                     labels=list(range(5)), zero_division=0)):
        rep[f"f1_class_{c}"] = float(f1c)
    rep.update(onset_f1(true_cls, pred_cls, prev_true_cls))
    return rep


def regression_report(true_ci, pred_ci, prev_true_ci) -> dict:
    err = pred_ci - true_ci
    rep = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
    # derived onset metrics via CI binning so cls and reg models are comparable
    rep.update(onset_f1(ci_to_cls(true_ci), ci_to_cls(np.clip(pred_ci, 0, 1)),
                        ci_to_cls(prev_true_ci)))
    return rep

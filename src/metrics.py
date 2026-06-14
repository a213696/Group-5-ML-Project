"""
src/metrics.py — Evaluation helpers for all three model stages.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def compute_metrics(y_true, y_pred, model_name="model"):
    return {
        "model":       model_name,
        "accuracy":    accuracy_score(y_true, y_pred),
        "macro_f1":    f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def classification_report_df(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True)
    return pd.DataFrame(report).T.round(4)


def confusion_matrix_array(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix_array(y_true, y_pred)
    return cm.diagonal() / cm.sum(axis=1)


def results_table(records):
    """
    records: list of dicts from compute_metrics()
    Returns a sorted DataFrame for the master comparison table.
    """
    df = pd.DataFrame(records)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df

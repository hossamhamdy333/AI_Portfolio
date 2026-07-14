
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)


def compute_metrics(y_true, y_pred, y_prob, label_names, average="weighted"):
    metrics = {
        "f1"       : f1_score(y_true, y_pred, average=average),
        "f1_macro" : f1_score(y_true, y_pred, average="macro"),
        "auc_roc"  : roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
    }
    print(classification_report(y_true, y_pred, target_names=label_names))
    return metrics


def plot_confusion_matrix(y_true, y_pred, label_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def error_analysis(df, y_true, y_pred, label_names, n=5):
    errors = df.copy().reset_index(drop=True)
    errors["true_label"]      = [label_names[i] for i in y_true]
    errors["predicted_label"] = [label_names[i] for i in y_pred]
    errors["correct"]         = (np.array(y_true) == np.array(y_pred))
    errors["confidence_gap"]  = abs(np.array(y_true) - np.array(y_pred))
    failures = errors[~errors["correct"]].sort_values("confidence_gap", ascending=False)
    print(f"Total errors: {len(failures)} / {len(df)} ({len(failures)/len(df)*100:.1f}%)")
    print(failures[["text", "true_label", "predicted_label"]].head(n).to_string())
    return failures

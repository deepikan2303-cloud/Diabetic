"""Metrics and plots: confusion matrix, F1, kappa, ROC."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                        digits=3, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
    if len(class_names) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probs[:, 1]) if len(np.unique(y_true)) == 2 else float("nan")
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
        metrics["sensitivity"] = tp / (tp + fn) if tp + fn else 0.0
        metrics["specificity"] = tn / (tn + fp) if tn + fp else 0.0
    else:
        # Quadratic weighted kappa: the official APTOS metric, penalises far-off grades more
        metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out: Path) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 3, 1.2 * len(class_names) + 2.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title="Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, scores: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, scores):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title="ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_history(history: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set(title=key.capitalize(), xlabel="Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

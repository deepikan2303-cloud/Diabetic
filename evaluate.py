"""Evaluate a trained run on its held-out test split.

    python evaluate.py --run outputs/binary_efficientnetb0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import make_dataset
from src.evaluation import compute_metrics, plot_confusion_matrix, plot_roc


def evaluate_run(run_dir: Path, split_name: str = "test") -> dict:
    cfg = json.loads((run_dir / "run_config.json").read_text())
    names = cfg["class_names"]
    df = pd.read_csv(run_dir / f"{split_name}.csv")
    model = tf.keras.models.load_model(run_dir / "model.keras")

    ds = make_dataset(df, len(names), training=False, size=cfg["img_size"])
    probs = model.predict(ds, verbose=0)
    y_true = df["label"].values
    m = compute_metrics(y_true, probs, names)

    lines = [
        f"Run: {run_dir.name}   split: {split_name}   n={len(df)}",
        f"Accuracy     : {m['accuracy']:.4f}",
        f"F1 (macro)   : {m['f1_macro']:.4f}",
        f"F1 (weighted): {m['f1_weighted']:.4f}",
    ]
    if "qwk" in m:
        lines.append(f"Quadratic weighted kappa: {m['qwk']:.4f}")
    else:
        lines += [f"ROC-AUC      : {m['roc_auc']:.4f}",
                  f"Sensitivity  : {m['sensitivity']:.4f}  (DR cases caught)",
                  f"Specificity  : {m['specificity']:.4f}  (healthy eyes correctly cleared)"]
    lines += ["", m["report"], "Confusion matrix (rows = true, cols = predicted):", str(m["confusion_matrix"])]
    text = "\n".join(lines)
    print(text)

    (run_dir / f"metrics_{split_name}.txt").write_text(text)
    plot_confusion_matrix(m["confusion_matrix"], names, run_dir / f"confusion_matrix_{split_name}.png")
    if len(names) == 2 and len(np.unique(y_true)) == 2:
        plot_roc(y_true, probs[:, 1], run_dir / f"roc_{split_name}.png")
    pd.DataFrame(probs, columns=[f"p_{n}" for n in names]).assign(
        id_code=df["id_code"], true=y_true, pred=probs.argmax(1)
    ).to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
    print(f"\nSaved metrics and plots to {run_dir}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder created by train.py")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()
    evaluate_run(args.run, args.split)


if __name__ == "__main__":
    main()

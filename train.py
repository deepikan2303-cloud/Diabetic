"""Train the DR classifier with two-phase transfer learning, then evaluate on the held-out test set.

    python train.py --mode binary
    python train.py --mode multiclass --backbone resnet50v2
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src import config
from src.data import class_weights, load_labels, make_dataset, split
from src.evaluation import plot_history
from src.model import build_model, compile_model, unfreeze_top
from evaluate import evaluate_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["binary", "multiclass"], default="binary",
                    help="binary = No DR vs DR (screening); multiclass = 5 severity grades")
    ap.add_argument("--backbone", choices=["efficientnetb0", "resnet50v2", "densenet121"], default="efficientnetb0")
    ap.add_argument("--csv", type=Path, default=config.LABELS_CSV)
    ap.add_argument("--images", type=Path, default=config.PROCESSED_DIR, help="folder of preprocessed PNGs")
    ap.add_argument("--out", type=Path, default=None, help="run folder (default outputs/<mode>_<backbone>)")
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    ap.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    ap.add_argument("--unfreeze", type=int, default=config.UNFREEZE_LAYERS)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (random init)")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)
    run_dir = args.out or config.OUTPUT_DIR / f"{args.mode}_{args.backbone}"
    run_dir.mkdir(parents=True, exist_ok=True)
    names = config.class_names(args.mode)

    # ---- data -------------------------------------------------------------
    df = load_labels(args.csv, args.images, args.mode)
    train_df, val_df, test_df = split(df)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(run_dir / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):5d}  class counts {part['label'].value_counts().sort_index().tolist()}")

    weights = class_weights(train_df["label"].values)
    print("class weights:", {names[k]: round(v, 2) for k, v in weights.items()})
    train_ds = make_dataset(train_df, len(names), True, args.img_size, args.batch_size)
    val_ds = make_dataset(val_df, len(names), False, args.img_size, args.batch_size)

    # ---- model ------------------------------------------------------------
    model, base = build_model(len(names), args.backbone, args.img_size,
                              None if args.weights == "none" else args.weights)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(run_dir / "model.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(run_dir / "history.csv", append=True),
    ]
    (run_dir / "history.csv").unlink(missing_ok=True)

    print("\n=== Phase 1: training classification head (backbone frozen) ===")
    compile_model(model, config.HEAD_LR)
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
              class_weight=weights, callbacks=callbacks)

    if args.fine_tune_epochs > 0:
        print(f"\n=== Phase 2: fine-tuning top {args.unfreeze} backbone layers ===")
        unfreeze_top(base, args.unfreeze)
        compile_model(model, config.FINE_TUNE_LR)
        model.fit(train_ds, validation_data=val_ds,
                  epochs=args.head_epochs + args.fine_tune_epochs, initial_epoch=args.head_epochs,
                  class_weight=weights, callbacks=callbacks)

    plot_history(pd.read_csv(run_dir / "history.csv").to_dict("list"), run_dir / "training_curves.png")
    (run_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "backbone": args.backbone, "img_size": args.img_size,
        "class_names": names, "images": str(args.images),
    }, indent=2))

    print("\n=== Evaluating best checkpoint on test set ===")
    evaluate_run(run_dir)


if __name__ == "__main__":
    main()

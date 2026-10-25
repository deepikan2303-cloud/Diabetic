"""Screen one or more raw fundus images with a trained model.

    python predict.py --run outputs/binary_efficientnetb0 path/to/eye1.png path/to/eye2.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocess import load_image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder created by train.py")
    ap.add_argument("--no-enhance", action="store_true", help="use if the model was trained without enhancement")
    ap.add_argument("images", nargs="+", type=Path)
    args = ap.parse_args()

    cfg = json.loads((args.run / "run_config.json").read_text())
    names = cfg["class_names"]
    model = tf.keras.models.load_model(args.run / "model.keras")

    batch = np.stack([load_image(p, cfg["img_size"], enhance=not args.no_enhance) for p in args.images])
    probs = model.predict(batch.astype("float32"), verbose=0)
    for path, p in zip(args.images, probs):
        k = int(p.argmax())
        detail = ", ".join(f"{n}: {v:.1%}" for n, v in zip(names, p))
        print(f"{path.name}: {names[k]} ({p[k]:.1%})   [{detail}]")


if __name__ == "__main__":
    main()

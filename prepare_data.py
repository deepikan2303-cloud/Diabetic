"""Preprocess every raw fundus image once and cache it as a PNG.

    python prepare_data.py --csv data/train.csv --images data/train_images --out data/processed
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

from src import config
from src.data import _find_image
from src.preprocess import load_image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=config.LABELS_CSV)
    ap.add_argument("--images", type=Path, default=config.RAW_IMAGE_DIR)
    ap.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--no-enhance", action="store_true", help="skip Ben Graham enhancement")
    args = ap.parse_args()

    if not args.csv.exists() or not args.images.is_dir():
        raise SystemExit(
            f"Dataset not found.\n  expected labels : {args.csv}\n  expected images : {args.images}\n"
            "Download APTOS 2019 from https://www.kaggle.com/competitions/aptos2019-blindness-detection/data\n"
            "and extract train.csv + train_images/ into the data folder (see README step 2),\n"
            "or pass --csv / --images to point at another location."
        )
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"{len(df)} images, grade distribution:\n{df['diagnosis'].value_counts().sort_index()}")

    skipped = 0
    for i, id_code in enumerate(df["id_code"].astype(str), 1):
        dst = args.out / f"{id_code}.png"
        if dst.exists():
            continue
        try:
            img = load_image(_find_image(args.images, id_code), args.size, enhance=not args.no_enhance)
        except FileNotFoundError as e:
            print(f"  skip: {e}")
            skipped += 1
            continue
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i % 200 == 0:
            print(f"  {i}/{len(df)}")
    print(f"Done. Processed images in {args.out} ({skipped} skipped)")


if __name__ == "__main__":
    main()

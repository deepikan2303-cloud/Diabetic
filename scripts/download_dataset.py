"""Download APTOS 2019 (all 3,662 labelled images, 224x224) from Hugging Face - no Kaggle login needed -
and unpack it into the layout the project expects:

    data/train.csv          (id_code, diagnosis)
    data/train_images/*.png

    python scripts/download_dataset.py

Source: https://huggingface.co/datasets/bumbledeep/aptos (MIT, resized copy of the Kaggle APTOS 2019 train set)
"""
import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

PARQUET_URL = "https://huggingface.co/datasets/bumbledeep/aptos/resolve/main/data/train-00000-of-00001.parquet"
ROOT = Path(__file__).resolve().parent.parent


def download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"Already downloaded: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        print(f"\r  {min(done, total) / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    tmp.rename(dst)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    parquet = args.out / "raw" / "aptos_train.parquet"
    download(PARQUET_URL, parquet)

    df = pd.read_parquet(parquet)
    img_dir = args.out / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        id_code = Path(row["image"].get("path") or f"aptos_{i:05d}").stem
        Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(img_dir / f"{id_code}.png")
        rows.append({"id_code": id_code, "diagnosis": int(row["label_code"])})
    labels = pd.DataFrame(rows)
    labels.to_csv(args.out / "train.csv", index=False)

    print(f"Wrote {len(labels)} images to {img_dir}")
    print(f"Labels: {args.out / 'train.csv'}")
    print("Grade counts:", labels["diagnosis"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

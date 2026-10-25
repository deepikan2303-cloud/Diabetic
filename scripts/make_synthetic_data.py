"""Generate a small synthetic fundus-like dataset in APTOS format for smoke-testing the pipeline.

The images are NOT medical data: an orange disc on black with vessel-like lines, plus
red dots / yellow spots whose count grows with the fake "grade". Use it only to check
that preprocessing, training and evaluation run end to end before downloading real data.

    python scripts/make_synthetic_data.py --n 300 --out data/synthetic
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def fake_fundus(grade: int, rng: np.random.Generator, size: int = 384) -> np.ndarray:
    img = np.zeros((size, size + 60, 3), np.uint8)  # wider than tall, like real fundus photos
    c = (img.shape[1] // 2, size // 2)
    r = size // 2 - 10
    base = np.array([20, 70, 170]) + rng.integers(-15, 15, 3)  # BGR orange-red
    cv2.circle(img, c, r, base.tolist(), -1)
    cv2.circle(img, (c[0] + r // 3, c[1]), r // 7, (120, 190, 240), -1)  # optic disc
    for _ in range(8):  # vessels
        angle = rng.uniform(0, 2 * np.pi)
        end = (int(c[0] + r // 3 + np.cos(angle) * r * 0.9), int(c[1] + np.sin(angle) * r * 0.9))
        cv2.line(img, (c[0] + r // 3, c[1]), end, (15, 30, 110), int(rng.integers(2, 5)))
    for _ in range(grade * 12):  # microaneurysms / haemorrhages
        p = c + (rng.uniform(-0.6, 0.6, 2) * r).astype(int)
        cv2.circle(img, tuple(int(v) for v in p), int(rng.integers(2, 5 + grade)), (10, 10, 120), -1)
    for _ in range(max(0, grade - 1) * 8):  # hard exudates
        p = c + (rng.uniform(-0.6, 0.6, 2) * r).astype(int)
        cv2.circle(img, tuple(int(v) for v in p), int(rng.integers(2, 5)), (90, 220, 240), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    img_dir = args.out / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # Imbalanced like APTOS: ~49% grade 0, few severe/proliferative cases
    grades = rng.choice(5, size=args.n, p=[0.49, 0.10, 0.27, 0.05, 0.09])
    rows = []
    for i, g in enumerate(grades):
        id_code = f"syn{i:05d}"
        cv2.imwrite(str(img_dir / f"{id_code}.png"), fake_fundus(int(g), rng))
        rows.append({"id_code": id_code, "diagnosis": int(g)})
    pd.DataFrame(rows).to_csv(args.out / "train.csv", index=False)
    print(f"Wrote {args.n} images to {img_dir} and labels to {args.out / 'train.csv'}")


if __name__ == "__main__":
    main()

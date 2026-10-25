"""Dataset loading, splitting and tf.data pipelines."""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from . import config


def load_labels(csv_path: Path, image_dir: Path, mode: str) -> pd.DataFrame:
    """Read the APTOS-style CSV and attach image paths and the training label."""
    df = pd.read_csv(csv_path)
    df["path"] = df["id_code"].astype(str).map(lambda i: str(_find_image(image_dir, i)))
    df["label"] = (df["diagnosis"] > 0).astype(int) if mode == "binary" else df["diagnosis"].astype(int)
    return df


def _find_image(image_dir: Path, id_code: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        p = image_dir / f"{id_code}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for id_code '{id_code}' in {image_dir}")


def split(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified 70 / 15 / 15 train / val / test split."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Balanced weights so rare grades (Severe, Proliferative) are not ignored."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.5, fill_mode="constant"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="constant"),
    tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
    tf.keras.layers.RandomContrast(0.1),
], name="augment")


def _read(path, label, size):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (size, size))
    return tf.cast(img, tf.float32), label


def make_dataset(df: pd.DataFrame, num_classes: int, training: bool,
                 size: int = config.IMG_SIZE, batch_size: int = config.BATCH_SIZE) -> tf.data.Dataset:
    """Build a pipeline over *preprocessed* PNGs (see prepare_data.py)."""
    labels = tf.one_hot(df["label"].values, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, labels))
    if training:
        ds = ds.shuffle(len(df), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _read(p, y, size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        ds = ds.map(lambda x, y: (_augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

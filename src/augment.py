"""
src/augment.py — Image augmentation pipeline for Stage 0b / Stage C.

LEAKAGE GUARDRAIL: Augmentation layers live INSIDE the Keras model
(as the first Sequential block). Keras automatically sets training=False
at inference time, so test data is NEVER augmented.
"""
import tensorflow as tf


def build_augmentation_layers():
    """
    Returns a Keras Sequential block of augmentation layers.
    Insert as the FIRST block in any CNN model.

    Transforms applied (training only):
    - RandomFlip("horizontal")  p=0.5  — most classes are symmetric
    - ZeroPadding2D(4) → RandomCrop(32,32) — translation invariance
    - RandomBrightness(0.1) / RandomContrast(0.1) — lighting robustness
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.ZeroPadding2D(padding=4),
        tf.keras.layers.RandomCrop(32, 32),
        tf.keras.layers.RandomBrightness(factor=0.1),
        tf.keras.layers.RandomContrast(factor=0.1),
    ], name="augmentation")


def preview_augmentation(x_batch, n=8, seed=42):
    """
    Show n original + n augmented images side by side.
    Returns (originals, augmented) numpy arrays.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_batch), n, replace=False)
    originals = x_batch[idx]

    aug = build_augmentation_layers()
    # Build the layer on a sample call
    aug_imgs = aug(originals.astype("float32") / 255.0, training=True).numpy()
    aug_imgs = (aug_imgs * 255).clip(0, 255).astype("uint8")
    return originals, aug_imgs

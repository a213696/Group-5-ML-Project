"""
src/data.py — Data loading, splitting, and preprocessing for D12 CIFAR-10.
All normalisation stats computed from TRAIN data only (no leakage).
"""
import os
import random
import hashlib
import numpy as np

SEED = 42
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def load_cifar10():
    from tensorflow.keras.datasets import cifar10
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()
    return x_train_full, y_train_full, x_test, y_test


def split_train_val(x_train_full, y_train_full, val_size=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train_full))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return (x_train_full[train_idx], y_train_full[train_idx],
            x_train_full[val_idx], y_train_full[val_idx])


def scale_to_float(x):
    return x.astype(np.float32) / 255.0


def compute_channel_stats(x_train_float):
    """Compute per-channel mean and std from TRAIN data only."""
    mean = x_train_float.mean(axis=(0, 1, 2))  # shape (3,)
    std  = x_train_float.std(axis=(0, 1, 2))   # shape (3,)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)


def flatten(x):
    return x.reshape(len(x), -1)


def hash_image(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def check_duplicates(x_train, x_test):
    train_hashes = {hash_image(img) for img in x_train}
    test_hashes  = {hash_image(img) for img in x_test}
    cross = train_hashes & test_hashes
    return len(train_hashes), len(test_hashes), len(cross)


def get_preprocessed_flat(seed=SEED):
    """
    Full preprocessing for Stages A & B.
    Returns flattened, z-score normalised arrays.
    Normalisation stats fitted on train only.
    """
    set_seeds(seed)
    x_train_full, y_train_full, x_test, y_test = load_cifar10()
    x_train, y_train, x_val, y_val = split_train_val(
        x_train_full, y_train_full, seed=seed)

    x_train_f = scale_to_float(x_train)
    x_val_f   = scale_to_float(x_val)
    x_test_f  = scale_to_float(x_test)

    mean, std = compute_channel_stats(x_train_f)   # train only!

    X_train = flatten(normalize(x_train_f, mean, std))
    X_val   = flatten(normalize(x_val_f,   mean, std))
    X_test  = flatten(normalize(x_test_f,  mean, std))

    stats = {"mean": mean, "std": std}
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), stats


def get_image_arrays(seed=SEED):
    """
    Returns raw uint8 image arrays + canonical split.
    Used by EDA notebook and Stage C (CNN takes its own preprocessing).
    """
    set_seeds(seed)
    x_train_full, y_train_full, x_test, y_test = load_cifar10()
    x_train, y_train, x_val, y_val = split_train_val(
        x_train_full, y_train_full, seed=seed)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

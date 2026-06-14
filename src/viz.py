"""
src/viz.py — Plotting helpers. All figures saved to artifacts/figures/ at 150 dpi.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")


# ------------------------------------------------------------------ #
# EDA plots
# ------------------------------------------------------------------ #

def plot_sample_grid(x, y, n_per_class=10, seed=42, save=True):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.2, 13))
    for cls in range(10):
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, n_per_class, replace=False)
        for j, img_idx in enumerate(chosen):
            img = x[img_idx]
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            axes[cls, j].imshow(img)
            axes[cls, j].axis("off")
        axes[cls, 0].set_ylabel(CLASSES[cls], fontsize=9,
                                 rotation=0, labelpad=45, va="center")
    fig.suptitle("CIFAR-10 — Sample Grid (10 images per class)\n"
                 "→ Intra-class variability is high; cat/dog/deer/horse visually overlap",
                 fontsize=11, y=1.005)
    plt.tight_layout()
    if save:
        _save("eda_01_sample_grid.png")
    plt.show()


def plot_class_distribution(y_train, y_test, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, y, title in zip(axes,
                              [y_train, y_test],
                              ["Train set (50,000 images)", "Test set (10,000 images)"]):
        counts = np.bincount(y, minlength=10)
        bars = ax.bar(CLASSES, counts, color="steelblue", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticklabels(CLASSES, rotation=40, ha="right")
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 50, str(v), ha="center", fontsize=8)
    fig.suptitle("Class Distribution — perfectly balanced (imbalance ratio = 1.0)\n"
                 "→ SMOTE/oversampling not required; image augmentation is the Stage 0b technique",
                 fontsize=10, y=1.03)
    plt.tight_layout()
    if save:
        _save("eda_02_class_distribution.png")
    plt.show()


def plot_channel_histograms(x_train_float, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    names  = ["Red", "Green", "Blue"]
    for c, (ax, color, name) in enumerate(zip(axes, colors, names)):
        vals = x_train_float[:, :, :, c].flatten()
        ax.hist(vals, bins=100, color=color, alpha=0.75, density=True)
        ax.axvline(vals.mean(), color="black", ls="--", lw=1.5,
                   label=f"mean={vals.mean():.3f}")
        ax.axvline(vals.mean() + vals.std(), color="gray", ls=":", lw=1,
                   label=f"std={vals.std():.3f}")
        ax.set_title(f"{name} Channel")
        ax.set_xlabel("Pixel value (scaled [0,1])")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("Per-channel Pixel Intensity Distributions (train set)\n"
                 "→ R/G/B channels differ in mean and spread → per-channel z-score normalisation required",
                 fontsize=10, y=1.03)
    plt.tight_layout()
    if save:
        _save("eda_03_channel_histograms.png")
    plt.show()


def plot_mean_images(x_train_float, y_train, save=True):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for cls in range(10):
        mask = y_train == cls
        mean_img = x_train_float[mask].mean(axis=0)
        axes[cls].imshow(mean_img.clip(0, 1))
        axes[cls].set_title(CLASSES[cls], fontsize=10)
        axes[cls].axis("off")
    fig.suptitle("Per-class Mean Image (train set)\n"
                 "→ Ships/frogs have distinct colour signatures; cat/dog/horse/deer overlap\n"
                 "→ Predicts LogReg confusion patterns",
                 fontsize=9, y=1.03)
    plt.tight_layout()
    if save:
        _save("eda_04_mean_images.png")
    plt.show()


# ------------------------------------------------------------------ #
# Model output plots
# ------------------------------------------------------------------ #

def plot_loss_curves(history, title="Loss Curves", save_name=None, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    keys  = [("train_loss", "val_loss"), ("train_acc", "val_acc")]
    ylabs = ["Loss", "Accuracy"]
    for ax, (tk, vk), ylab in zip(axes, keys, ylabs):
        ax.plot(history[tk], label="Train")
        ax.plot(history[vk], label="Val", ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title} — {ylab}")
        ax.legend()
    plt.tight_layout()
    if save and save_name:
        _save(save_name)
    plt.show()


def plot_confusion_matrix(cm, title="Confusion Matrix", save_name=None, save=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.3)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save and save_name:
        _save(save_name)
    plt.show()


def plot_comparison_bar(results_df, metric="accuracy", save=True):
    fig, ax = plt.subplots(figsize=(12, 5))
    vals = results_df[metric].values * 100
    bars = ax.bar(results_df["model"], vals, color="steelblue", edgecolor="white")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()} (CIFAR-10 test set)")
    ax.set_ylim(0, 100)
    ax.axhline(10, color="red", ls="--", lw=1, label="Random baseline (10%)")
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save:
        _save(f"results_{metric}_bar.png")
    plt.show()


def plot_per_class_accuracy_heatmap(model_pca_dict, save=True):
    """
    model_pca_dict: {"Model Name": per_class_acc_array (10,), ...}
    """
    import pandas as pd
    df = pd.DataFrame(model_pca_dict, index=CLASSES).T
    fig, ax = plt.subplots(figsize=(14, len(model_pca_dict) * 0.9 + 1))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.3)
    ax.set_title("Per-class Accuracy by Model\n→ Reveals where each model fails")
    ax.set_xlabel("Class")
    ax.set_ylabel("Model")
    plt.tight_layout()
    if save:
        _save("results_per_class_heatmap.png")
    plt.show()


def plot_depth_vs_accuracy(depths, accuracies, save=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, [a * 100 for a in accuracies],
            marker="o", linewidth=2, markersize=8, color="steelblue")
    for d, a in zip(depths, accuracies):
        ax.annotate(f"{a*100:.1f}%", (d, a * 100),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("Number of Conv Blocks")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("CNN Depth vs. Test Accuracy (D12 Key Study)\n"
                 "→ Diminishing returns beyond 4 blocks on 32×32 images")
    ax.set_xticks(depths)
    plt.tight_layout()
    if save:
        _save("stageC_depth_vs_accuracy.png")
    plt.show()

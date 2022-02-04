import glob
import json
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

DATA_FOLDER = os.path.join(os.getcwd(), "data")
DATASET_FOLDER = os.path.join(os.getcwd(), "data", "raw", "dataset")
RETRIEVAL_FOLDER = os.path.join(os.getcwd(), "data", "raw", "retrieval")
FEATURES_FOLDER = os.path.join(os.getcwd(), "data", "features")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "out")

STANDARD_FEATURES_SIZE = (512, 512)

# For semplicity let ccv the last (not mandatory)
LIST_OF_FEATURES_IMPLEMENTED = [
    "rgb_hist",
    "hsv_hist",
    "lbp",
    "hog",
    "dct",
    "vgg",
    "resnet50",
    "ccv",
    "orb_desc",
    "orb_kps"
]


def load_features(feature):
    if feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    filepath = os.path.join(FEATURES_FOLDER, f"{feature}.npy")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"feature '{feature}' not computed")

    return joblib.load(filepath, mmap_mode="r+")

def plot_image(image):
    fig, axes = plt.subplots()

    # hide axis
    for ax in axes.ravel():
        ax.set_axis_off()

    # query image
    axes[0, 2].imshow(
        cv.cvtColor(image, cv.COLOR_BGR2RGB)
    )
    axes[0, 2].set_title("Image")

    plt.show()


if __name__ == "__main__":
    F = load_features("rgb_hist")
    print(F.shape)


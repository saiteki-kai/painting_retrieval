import os

import cv2 as cv
import joblib
import matplotlib.pyplot as plt

from src.config import LIST_OF_FEATURES_IMPLEMENTED, FEATURES_FOLDER


def load_features(feature):
    if feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    filepath = os.path.join(FEATURES_FOLDER, f"{feature}.npy")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"feature '{feature}' not computed")

    return joblib.load(filepath, mmap_mode="r+")


def plot_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # displaying image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

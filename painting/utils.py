import os.path

import numpy as np

TRAIN_FOLDER = os.path.join(os.getcwd(), "data", "raw", "train")
TEST_FOLDER = os.path.join(os.getcwd(), "data", "raw", "test")
FEATURES_FOLDER = os.path.join(os.getcwd(), "data", "features")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "out")


def load_features(feature):
    if feature not in ["hog", "hsv_hist", "rgb_hist", "bow_sift"]:
        raise ValueError(f"unrecognized feature: '{feature}'")

    filepath = os.path.join(FEATURES_FOLDER, f"{feature}.npy")

    if not os.path.exists(filepath):
        raise IOError(f"feature '{feature}' not saved")

    return np.load(filepath)

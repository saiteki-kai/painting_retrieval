import os.path

import numpy as np

TRAIN_FOLDER = os.path.join(os.getcwd(), "data", "raw", "train")
TEST_FOLDER = os.path.join(os.getcwd(), "data", "raw", "test")
RETRIEVAL_FOLDER = os.path.join(os.getcwd(), "data", "raw", "retrieval")
FEATURES_FOLDER = os.path.join(os.getcwd(), "data", "features")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "out")


def load_features(feature):
    list_of_features = [
        "hog", "hsv_hist", "lbp", 
        "rgb_hist", "bow_sift",
        "ccv", "dct"]
        
    if feature not in list_of_features:
        raise ValueError(f"unrecognized feature: '{feature}'")

    filepath = os.path.join(FEATURES_FOLDER, f"{feature}.npy")

    if not os.path.exists(filepath):
        raise IOError(f"feature '{feature}' not saved")

    return np.load(filepath)

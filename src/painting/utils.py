import os

import cv2 as cv
import joblib
import matplotlib.pyplot as plt

#BASE_FOLDER = os.path.dirname(os.getcwd())
BASE_FOLDER = os.getcwd()
DATA_FOLDER = os.path.join(BASE_FOLDER, "data")
DATASET_FOLDER = os.path.join(BASE_FOLDER, "data", "raw", "dataset")
RETRIEVAL_FOLDER = os.path.join(BASE_FOLDER, "data", "raw", "retrieval")
FEATURES_FOLDER = os.path.join(BASE_FOLDER, "data", "features")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "out")
MODEL_FOLDER = os.path.join(BASE_FOLDER, "model")


STANDARD_FEATURES_SIZE = (512, 512)

# For semplicity let ccv the last (not mandatory)
LIST_OF_FEATURES_IMPLEMENTED = [
    "rgb_hist",
    "hsv_hist",
    "lbp",
    "hog",
    "dct",
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

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # displaying image
    plt.imshow(image)
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    F = load_features("rgb_hist")
    print(F.shape)

import os

BASE_FOLDER = os.path.abspath(".")
DATA_FOLDER = os.path.join(BASE_FOLDER, "data")
DATASET_FOLDER = os.path.join(BASE_FOLDER, "data", "raw", "dataset")
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
    "orb"
]

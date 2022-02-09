import os
import cv2 as cv
import numpy as np

from sklearn.decomposition import PCA
from joblib import dump

from src.config import DATASET_FOLDER, LIST_OF_FEATURES_IMPLEMENTED, STANDARD_FEATURES_SIZE, FEATURES_FOLDER
from src.painting.dataset import Dataset
from src.painting.extract import compute_descriptor
from src.painting.exact_matching import compute_orb_descriptor
from src.painting.utils import resize_with_max_ratio, load_features


def combine_features():
    rgb_hist = load_features("rgb_hist")
    lbp_gray = load_features("lbp")

    combined = np.hstack((rgb_hist, lbp_gray))
    print(combined.shape)

    pca = PCA(0.99)
    combined = pca.fit_transform(combined)
    print(combined.shape)

    dump(combined, os.path.join(FEATURES_FOLDER, "combined.npy"))
    dump(pca, os.path.join(FEATURES_FOLDER, "pca_params"), compress=True)


if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    # We avoid computing ccv for now (too slow)
    avoid_list = ['ccv', 'resnet50']
    list_of_features = [x for x in LIST_OF_FEATURES_IMPLEMENTED if x not in avoid_list]

    # list_of_features = []
    for feature in list_of_features:
        print("Computing: " + feature)
        compute_descriptor(ds, feature)

    combine_features()

    # ------------------------------------------------------------------------------------------------------------------

    # We want to compute resnet now to observe better the results.
    ds = Dataset(DATASET_FOLDER, (224, 224))

    print("Computing: resnet50")
    compute_descriptor(ds, "resnet50")

    # ------------------------------------------------------------------------------------------------------------------

    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    print("Computing: ORB")
    compute_orb_descriptor(ds)

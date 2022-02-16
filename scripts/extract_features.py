import os
import cv2 as cv
import numpy as np

from sklearn.decomposition import PCA
from joblib import dump

from src.config import DATASET_FOLDER, LIST_OF_FEATURES_IMPLEMENTED, STANDARD_FEATURES_SIZE, FEATURES_FOLDER
from src.painting.dataset import Dataset
from src.painting.extract import compute_descriptor
from src.painting.utils import load_features, resize_with_max_ratio


def combine_features():
    lbp, lbp_scaler = load_features("lbp", return_scaler=True)
    hist, hist_scaler = load_features("local_rgb_hist", return_scaler=True)
    hog, hog_scaler = load_features("hog", return_scaler=True)

    lbp = lbp_scaler.transform(lbp)
    hist = hist_scaler.transform(hist)
    hog = hog_scaler.transform(hog)

    combined = np.hstack((lbp, hist, hog))
    print(combined.shape)

    pca = PCA(0.9)
    combined = pca.fit_transform(combined)
    print(combined.shape)

    dump(combined, os.path.join(FEATURES_FOLDER, "combined.npy"))
    dump(pca, os.path.join(FEATURES_FOLDER, "pca_params"), compress=True)


if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    avoid_list = ['resnet50', 'sift']
    list_of_features = [x for x in LIST_OF_FEATURES_IMPLEMENTED if x not in avoid_list]

    for feature in list_of_features:
        print("Computing: " + feature)
        compute_descriptor(ds, feature)

    combine_features()

    # ------------------------------------------------------------------------------------------------------------------

    # We want to compute resnet now to observe better the results.
    ds = Dataset(DATASET_FOLDER, (224, 224))

    print("Computing: resnet50")
    compute_descriptor(ds, "resnet50")

    # SIFT -------------------------------------------------------------------------------------------------------------

    def custom_read(filepath):
        img = cv.imread(filepath)
        img = resize_with_max_ratio(img, 512, 512)
        return img


    ds = Dataset(DATASET_FOLDER, image_size=None, custom_read_image=custom_read)
    compute_descriptor(ds, 'sift')

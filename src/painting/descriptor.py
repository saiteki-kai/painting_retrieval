import os.path

import cv2 as cv
import numpy as np
from joblib import load
from skimage.feature import hog

from src.config import LIST_OF_FEATURES_IMPLEMENTED, FEATURES_FOLDER
from src.painting.exact_matching import compute_sift
from src.painting.features.edges import local_edge_hist
from src.painting.features.histogram import compute_rgb_hist, compute_local_rgb_hist, compute_hsv_hist, \
    compute_local_hsv_hist
from src.painting.features.lbp import compute_lbp_gray
from src.painting.features.resnet import get_resnet50
from src.painting.features.bow import featuresBOW
from src.painting.utils import load_scaler


def compute_feature(img, feature):
    if feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    if feature == "rgb_hist":
        return compute_rgb_hist(img, bins=(128, 128, 128))
    if feature == "local_rgb_hist":
        return compute_local_rgb_hist(img, block_size=64, bins=(16, 16, 16))
    elif feature == "hsv_hist":
        return compute_hsv_hist(img, bins=(16, 16, 16))
    elif feature == "local_hsv_hist":
        return compute_local_hsv_hist(img, block_size=128, bins=(16, 16, 16))
    elif feature == "edge_hist":
        return local_edge_hist(img)
    elif feature == "lbp":
        return compute_lbp_gray(img, radius=1)
    elif feature == "hog":
        return compute_hog(img)
    elif feature == "resnet50":
        return compute_resnet50(img)
    elif feature == "sift":
        return compute_sift(img, dense=False)
    elif feature == "bow":
        return compute_bow(img)
    elif feature == "combined":
        return compute_combined(img)


def compute_hog(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    H = hog(img, pixels_per_cell=(128, 128), cells_per_block=(2, 2))
    return H


def compute_resnet50(dataset):
    return get_resnet50(dataset=dataset)


def compute_bow(img):
    return np.asarray(featuresBOW(img))


def compute_combined(img):
    lbp = compute_feature(img, "lbp")
    hist = compute_feature(img, "local_rgb_hist")
    h = compute_feature(img, "hog")

    lbp_scaler = load_scaler("lbp")
    hist_scaler = load_scaler("local_rgb_hist")
    hog_scaler = load_scaler("hog")

    lbp = lbp_scaler.transform(lbp.reshape(1, -1))
    hist = hist_scaler.transform(hist.reshape(1, -1))
    h = hog_scaler.transform(h.reshape(1, -1))

    # resnet = get_resnet50(img)
    # resnet = resnet / (np.linalg.norm(resnet, 2, axis=0) + 1e-7)

    combined = np.hstack((lbp, hist, h))
    pca = load(os.path.join(FEATURES_FOLDER, "pca_params"))

    combined = pca.transform(combined.reshape(1, -1))
    return combined

import os.path

import cv2 as cv
import numpy as np
from joblib import load
from skimage.feature import hog

from src.config import LIST_OF_FEATURES_IMPLEMENTED, FEATURES_FOLDER
from src.painting.features.edges import local_edge_hist
from src.painting.features.histogram import compute_rgb_hist, compute_local_rgb_hist, compute_hsv_hist, \
    compute_local_hsv_hist
from src.painting.features.lbp import compute_lbp_rgb, compute_lbp_gray
from src.painting.features.resnet import get_resnet50


def compute_feature(img, feature):
    if feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    if feature == "rgb_hist":
        return compute_rgb_hist(img, bins=32)
    if feature == "local_rgb_hist":
        return compute_local_rgb_hist(img, block_size=128, bins=256)
    elif feature == "hsv_hist":
        return compute_hsv_hist(img, bins=(16, 4, 4))  # 180 256, 256
    elif feature == "local_hsv_hist":
        return compute_local_hsv_hist(img, block_size=128, bins=(180, 256, 256))
    elif feature == "edge_hist":
        return local_edge_hist(img)
    elif feature == "lbp":
        return compute_lbp_rgb(img, radius=3)
    elif feature == "hog":
        return compute_hog(img)
    elif feature == "resnet50":
        return compute_resnet50(img)
    elif feature == "orb":
        return compute_orb(img)
    elif feature == "combined":
        return compute_combined(img)


def compute_orb(img, n_features=500):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(gray, None)
    kp = np.asarray(kp)
    des = np.asarray(des)

    print(kp.shape, des.shape)
    return kp, des


def compute_hog(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    H = hog(img, pixels_per_cell=(64, 64), cells_per_block=(4, 4))
    return H


def compute_resnet50(dataset):
    return get_resnet50(dataset=dataset)


def compute_combined(img):
    lbp_gray = compute_lbp_gray(img)
    edge_hist = local_edge_hist(img, block_size=128)
    rgb_hist = compute_rgb_hist(img, bins=8)
    resnet = get_resnet50(img)
    resnet = resnet / (np.linalg.norm(resnet, 2, axis=0) + 1e-7)

    combined = np.hstack((edge_hist, lbp_gray, rgb_hist, resnet))
    pca = load(os.path.join(FEATURES_FOLDER, "pca_params"))

    combined = pca.transform(combined.reshape(1, -1))
    return combined

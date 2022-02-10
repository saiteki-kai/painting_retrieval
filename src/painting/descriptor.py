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
from src.painting.features.bow import featuresBOW


def compute_feature(img, feature):
    if feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    if feature == "rgb_hist":
        return compute_rgb_hist(img, bins=128)
    if feature == "local_rgb_hist":
        return compute_local_rgb_hist(img, block_size=64, bins=(16, 16, 16))
    elif feature == "hsv_hist":
        return compute_hsv_hist(img, bins=(16, 16, 16))
    elif feature == "local_hsv_hist":
        return compute_local_hsv_hist(img, block_size=128, bins=(16, 16, 16))
    elif feature == "edge_hist":
        return local_edge_hist(img)
    elif feature == "lbp":
        return compute_lbp_rgb(img, radius=1)
    elif feature == "hog":
        return compute_hog(img)
    elif feature == "resnet50":
        return compute_resnet50(img)
    elif feature == "orb":
        return compute_orb(img)
    elif feature == "bow":
        return compute_bow(img)
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
    H = hog(img, pixels_per_cell=(128, 128), cells_per_block=(2, 2))
    return H


def compute_resnet50(dataset):
    return get_resnet50(dataset=dataset)

def compute_bow(img):
    return featuresBOW(img)


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

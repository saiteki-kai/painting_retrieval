import os.path

import cv2 as cv
import numpy as np
from joblib import load
from skimage.feature import local_binary_pattern, hog

from src.config import LIST_OF_FEATURES_IMPLEMENTED, FEATURES_FOLDER
from src.painting.ccv import get_ccv
from src.painting.features_extraction import get_resnet50


def compute_feature(img, feature):
    if feature != "combined" and feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    if feature == "rgb_hist":
        return compute_rgb_hist(img, bins=256)
    if feature == "local_rgb_hist":
        return compute_local_rgb_hist(img, block_size=256, bins=32)
    elif feature == "hsv_hist":
        return compute_hsv_hist(img, bins=(16, 4, 4))  # 180 256, 256
    elif feature == "local_hsv_hist":
        return compute_local_hsv_hist(img, block_size=256, bins=8)
    elif feature == "lbp":
        return compute_lbp_gray(img)
    elif feature == "hog":
        return compute_hog(img)
    elif feature == "dct":
        return compute_dct(img)
    elif feature == "resnet50":
        return compute_resnet50(img)
    elif feature == "ccv":
        return compute_ccv(img)
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


def compute_rgb_hist(img, bins=256):
    blue_hist = cv.calcHist([img], [0], None, [bins], [0, 256])
    green_hist = cv.calcHist([img], [1], None, [bins], [0, 256])
    red_hist = cv.calcHist([img], [2], None, [bins], [0, 256])

    blue_hist = cv.normalize(blue_hist, None, 0, 1, cv.NORM_MINMAX)
    green_hist = cv.normalize(green_hist, None, 0, 1, cv.NORM_MINMAX)
    red_hist = cv.normalize(red_hist, None, 0, 1, cv.NORM_MINMAX)

    return np.vstack([blue_hist, green_hist, red_hist]).ravel()


def compute_local_color_hist(img, block_size=128, bins=256):
    b = []
    for row in np.arange(0, img.shape[0], block_size):
        for col in np.arange(0, img.shape[1], block_size):
            block = img[row: row + block_size, col: col + block_size]
            b.append(compute_rgb_hist(block, bins))

    return np.array(b).flatten()


def compute_local_rgb_hist(img, block_size=128, bins=256):
    return compute_local_color_hist(img, block_size, bins)


def compute_local_hsv_hist(img, block_size=128, bins=256):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return compute_local_color_hist(img, block_size, bins)


def compute_hsv_hist(img, bins=(8, 8, 8)):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    hue_hist = cv.calcHist([img], [0], None, [bins[0]], [0, 180])
    sat_hist = cv.calcHist([img], [1], None, [bins[1]], [0, 256])
    val_hist = cv.calcHist([img], [2], None, [bins[2]], [0, 256])

    hue_hist = cv.normalize(hue_hist, None, 0, 1, cv.NORM_MINMAX)
    sat_hist = cv.normalize(sat_hist, None, 0, 1, cv.NORM_MINMAX)
    val_hist = cv.normalize(val_hist, None, 0, 1, cv.NORM_MINMAX)

    return np.vstack([hue_hist, sat_hist, val_hist]).ravel()


def compute_channel_lbp(img):
    radius = 1 # 2
    n_points = radius * 8

    lbp = local_binary_pattern(img, n_points, radius, "nri_uniform")

    bins = n_points * (n_points - 1) + 3
    lims = (0, n_points + 2)

    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=lims)
    hist = hist / (np.linalg.norm(hist) + 1e-7)
    return hist.ravel()


def compute_lbp_gray(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return compute_channel_lbp(img)


def compute_lbp_rgb(img):
    return np.concatenate(
        (
            compute_channel_lbp(img[:, :, 0]),
            compute_channel_lbp(img[:, :, 1]),
            compute_channel_lbp(img[:, :, 2]),
        )
    )


def compute_hog(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    H = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return H


def compute_dct(img):
    num_channels = img.shape[-1]

    if num_channels == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = np.float32(img) / 255.0
    dct = cv.dct(img)
    dct = np.uint8(dct * 255)

    return dct


def compute_resnet50(dataset):
    return get_resnet50(dataset=dataset)


def compute_ccv(img, n=2, tau=0.01):
    return get_ccv(img, n=n, tau=tau)


def compute_combined(img):
    lbp_gray = compute_lbp_gray(img)
    rgb_hist = compute_rgb_hist(img, bins=256)
    print(lbp_gray.shape)
    print(rgb_hist.shape)

    combined = np.hstack((rgb_hist, lbp_gray))
    pca = load(os.path.join(FEATURES_FOLDER, "pca_params"))

    combined = pca.transform(combined.reshape(1, -1))
    print(combined.shape)
    return combined

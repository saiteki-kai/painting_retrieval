import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern


def compute_channel_lbp(img, radius=1):
    n_points = radius * 8

    lbp = local_binary_pattern(img, n_points, radius, "nri_uniform")

    bins = n_points * (n_points - 1) + 3
    lims = (0, n_points + 2)

    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=lims)
    hist = hist / (np.linalg.norm(hist) + 1e-7)
    return hist.ravel()


def compute_lbp_gray(img, radius=1):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return compute_channel_lbp(img, radius)


def compute_lbp_rgb(img, radius=1):
    return np.concatenate(
        (
            compute_channel_lbp(img[:, :, 0], radius),
            compute_channel_lbp(img[:, :, 1], radius),
            compute_channel_lbp(img[:, :, 2], radius),
        )
    )

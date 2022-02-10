import cv2 as cv
import numpy as np


def compute_channel_hist(img, ranges=(0, 256), bins=256):
    hist = cv.calcHist([img], [0], None, [bins], ranges)
    hist = cv.normalize(hist, None, 0, 1, cv.NORM_MINMAX)
    return hist


def compute_rgb_hist(img, bins=(256, 256, 256)):
    b_hist = compute_channel_hist(img[:, :, 0], ranges=[0, 256], bins=bins[0])
    g_hist = compute_channel_hist(img[:, :, 1], ranges=[0, 256], bins=bins[1])
    r_hist = compute_channel_hist(img[:, :, 2], ranges=[0, 256], bins=bins[2])

    return np.vstack([b_hist, g_hist, r_hist]).ravel()


def compute_hsv_hist(img, bins=(180, 256, 256)):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    hue_hist = compute_channel_hist(img[:, :, 0], ranges=[0, 180], bins=bins[0])
    sat_hist = compute_channel_hist(img[:, :, 1], ranges=[0, 256], bins=bins[1])
    val_hist = compute_channel_hist(img[:, :, 2], ranges=[0, 256], bins=bins[2])

    return np.vstack([hue_hist, sat_hist, val_hist]).ravel()


def compute_local_hist(img, compute_hist_fn, block_size=128, bins=(256, 256, 256)):
    blocks = []
    for row in np.arange(0, img.shape[0], block_size):
        for col in np.arange(0, img.shape[1], block_size):
            block = img[row: row + block_size, col: col + block_size]
            blocks.append(compute_hist_fn(block, bins))

    return np.array(blocks).flatten()


def compute_local_rgb_hist(img, block_size=128, bins=(256, 256, 256)):
    return compute_local_hist(img, compute_rgb_hist, block_size, bins)


def compute_local_hsv_hist(img, block_size=128, bins=(180, 256, 256)):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return compute_local_hist(img, compute_hsv_hist, block_size, bins)

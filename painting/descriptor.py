import cv2 as cv
import numpy as np


class ImageDescriptor:

    @staticmethod
    def describe(img, feature):
        if feature not in ["hog", "hsv_hist", "rgb_hist", "bow_sift"]:
            raise ValueError(f"unrecognized feature: '{feature}'")

        if feature == "rgb_hist":
            return ImageDescriptor.compute_rgb_hist(img)
        elif feature == "hsv_hist":
            return ImageDescriptor.compute_hsv_hist(img)
        else:
            pass

    @staticmethod
    def compute_rgb_hist(img):
        blue_hist = cv.calcHist([img], [0], None, [256], [0, 256])
        green_hist = cv.calcHist([img], [1], None, [256], [0, 256])
        red_hist = cv.calcHist([img], [2], None, [256], [0, 256])

        blue_hist = cv.normalize(blue_hist, None, 0, 1, cv.NORM_MINMAX)
        green_hist = cv.normalize(green_hist, None, 0, 1, cv.NORM_MINMAX)
        red_hist = cv.normalize(red_hist, None, 0, 1, cv.NORM_MINMAX)

        return np.vstack([blue_hist, green_hist, red_hist]).ravel()

    @staticmethod
    def compute_hsv_hist(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        hue_hist = cv.calcHist([img], [0], None, [180], [0, 180])
        sat_hist = cv.calcHist([img], [1], None, [256], [0, 256])
        val_hist = cv.calcHist([img], [2], None, [256], [0, 256])

        hue_hist = cv.normalize(hue_hist, None, 0, 1, cv.NORM_MINMAX)
        sat_hist = cv.normalize(sat_hist, None, 0, 1, cv.NORM_MINMAX)
        val_hist = cv.normalize(val_hist, None, 0, 1, cv.NORM_MINMAX)

        return np.vstack([hue_hist, sat_hist, val_hist]).ravel()

import cv2 as cv
import numpy as np


def edge_hist(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gy = cv.Sobel(img, cv.CV_32F, 1, 0, None)
    gx = cv.Sobel(img, cv.CV_32F, 0, 1, None)

    gm = cv.magnitude(gx, gy)
    ga = cv.phase(gx, gy, angleInDegrees=True)

    gm_hist = cv.calcHist([gm], [0], None, [10], [0, 1448])
    gm_hist = cv.normalize(gm_hist, None, 0, 1, cv.NORM_MINMAX)

    ga_hist = cv.calcHist([ga], [0], None, [8], [0, 360])
    ga_hist = cv.normalize(ga_hist, None, 0, 1, cv.NORM_MINMAX)

    H = np.vstack((gm_hist, ga_hist))

    return H.ravel()


def local_edge_hist(img, block_size=128):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gy = cv.Sobel(img, cv.CV_32F, 1, 0, None)
    gx = cv.Sobel(img, cv.CV_32F, 0, 1, None)

    gm = cv.magnitude(gx, gy)
    ga = cv.phase(gx, gy, angleInDegrees=True)

    b = []
    for row in np.arange(0, img.shape[0], block_size):
        for col in np.arange(0, img.shape[1], block_size):
            block_m = gm[row: row + block_size, col: col + block_size]
            block_a = ga[row: row + block_size, col: col + block_size]

            gm_hist = cv.calcHist([block_m], [0], None, [10], [0, 1448])
            gm_hist = cv.normalize(gm_hist, None, 0, 1, cv.NORM_MINMAX)

            ga_hist = cv.calcHist([block_a], [0], None, [8], [0, 360])
            ga_hist = cv.normalize(ga_hist, None, 0, 1, cv.NORM_MINMAX)

            H = np.vstack((gm_hist, ga_hist))
            b.append(H)

    return np.array(b).flatten()

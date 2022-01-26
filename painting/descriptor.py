import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
from ccv import get_ccv

def compute_feature(img, feature):
    list_of_features = [
        "hog", "hsv_hist", "lbp", 
        "rgb_hist", "bow_sift",
        "ccv", "dct"]

    if feature not in list_of_features:
        raise ValueError(f"unrecognized feature: '{feature}'")

    if feature == "rgb_hist":
        return compute_rgb_hist(img)
    elif feature == "hsv_hist":
        return compute_hsv_hist(img)
    elif feature == "lbp":
        return compute_lbp(img)
    elif feature == "hog":
        return compute_hog(img)
    elif feature == "ccv":
        return compute_ccv(img)
    elif feature == "dct":
        return compute_dct(img)



def compute_rgb_hist(img):
    blue_hist = cv.calcHist([img], [0], None, [256], [0, 256])
    green_hist = cv.calcHist([img], [1], None, [256], [0, 256])
    red_hist = cv.calcHist([img], [2], None, [256], [0, 256])

    blue_hist = cv.normalize(blue_hist, None, 0, 1, cv.NORM_MINMAX)
    green_hist = cv.normalize(green_hist, None, 0, 1, cv.NORM_MINMAX)
    red_hist = cv.normalize(red_hist, None, 0, 1, cv.NORM_MINMAX)

    return np.vstack([blue_hist, green_hist, red_hist]).ravel()


def compute_hsv_hist(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    hue_hist = cv.calcHist([img], [0], None, [180], [0, 180])
    sat_hist = cv.calcHist([img], [1], None, [256], [0, 256])
    val_hist = cv.calcHist([img], [2], None, [256], [0, 256])

    hue_hist = cv.normalize(hue_hist, None, 0, 1, cv.NORM_MINMAX)
    sat_hist = cv.normalize(sat_hist, None, 0, 1, cv.NORM_MINMAX)
    val_hist = cv.normalize(val_hist, None, 0, 1, cv.NORM_MINMAX)

    return np.vstack([hue_hist, sat_hist, val_hist]).ravel()


def compute_lbp(img):
    radius = 2
    n_points = radius * 8
    #   cell_size = (512, 512)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img, n_points, radius, "uniform")

    bins = np.arange(0, n_points + 3)
    lims = (0, n_points + 2)

    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=lims)
    hist = hist / (np.linalg.norm(hist) + 1e-7)
    return hist.ravel()

    # w, h = lbp.shape

    # n_cell_x = int(w // cell_size[0])
    # n_cell_y = int(h // cell_size[1])
    # n_cells = np.prod(np.floor(lbp.shape / cell_size))

    # out = []
    # for i in range(0, w, cell_size[0]):
    #     for j in range(0, h, cell_size[1]):
    #         cell = lbp[i:(i + cell_size[0]), j:(j + cell_size[0])]
    #         hist, _ = np.histogram(cell.ravel(), bins=bins, range=lims)
    #         hist = hist / (np.linalg.norm(hist) + 1e-7)
    #         out.append(hist.ravel())
    #
    # return np.array(out).ravel()

def compute_hog(img):
    hog = cv.HOGDescriptor()
    img = hog.compute(img)

    return img

def compute_ccv(img, n=2, tau=0.01):
    return get_ccv(img, n=n, tau=tau) 

def compute_dct(img):
    num_channels = img.shape[-1]

    if num_channels == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = np.float32(img)/255.0
    dct = cv.dct(img)
    dct = np.uint8(dct * 255)

    return dct
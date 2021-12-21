import os

import cv2 as cv
import numpy as np

from painting.descriptor import ImageDescriptor
from painting.utils import TRAIN_FOLDER, FEATURES_FOLDER

if __name__ == "__main__":
    image_list = os.listdir(TRAIN_FOLDER)
    image_list = list(map(lambda fn: os.path.join(TRAIN_FOLDER, fn), image_list))

    N_images = len(image_list)

    features = np.zeros((N_images, 180 + 256 + 256))
    for idx, filename in enumerate(image_list):
        img = cv.imread(filename)
        h = ImageDescriptor.compute_hsv_hist(img)
        # print(f"{img.shape} -> {h.shape}")
        features[idx, :] = h

    print(len(features))
    np.save(os.path.join(FEATURES_FOLDER, "hsv_hist"), features)

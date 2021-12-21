import os

import cv2 as cv


class Dataset:
    def __init__(self, path):
        self.path = path

        self.image_list = os.listdir(self.path)
        self.image_list = list(map(lambda fn: os.path.join(self.path, fn), self.image_list))

    def get_image_by_index(self, index):
        return cv.imread(self.image_list[index])

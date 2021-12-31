import os
from tensorflow.keras.preprocessing import image as image_vgg
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2 as cv


class Dataset:
    def __init__(self, path, image_size=None):
        self.path = path

        self._image_list = os.listdir(self.path)
        self._image_list = list(map(lambda fn: os.path.join(self.path, fn), self._image_list))

        self._image_size = image_size

    def images(self):
        N = len(self._image_list)
        for i in range(N):
            yield self.get_image_by_index(i)

    def length(self):
        return len(self._image_list)

    def get_image_by_index(self, index):
        img = cv.imread(self._image_list[index])

        if self._image_size:
            return cv.resize(img, self._image_size)

        return img

    def get_image_by_filename(self, filename):
        matches = list(filter(lambda x: x.endswith(filename), self._image_list))

        if len(matches) == 0 or not os.path.exists(matches[0]):
            raise FileNotFoundError(f"Image {filename} not found")

        img = cv.imread(matches[0])

        if self._image_size:
            return cv.resize(img, self._image_size)

        return img

    def get_image_by_index_vgg(self, index):
        """
            VGG need different steps.
        """
        img = image_vgg.load_img( self._image_list[index], target_size=(224, 224) )
        img = image_vgg.img_to_array(img)
        img = np.array([img])
        #img = preprocess_input(img)

        return img

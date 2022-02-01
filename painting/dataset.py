import os
import glob
import cv2 as cv
import pandas as pd


class Dataset:
    def __init__(
        self, data: pd.DataFrame, folder, image_size=None, custom_read_image=None
    ):
        self._data = data
        self._folder = folder
        self._image_size = image_size
        self._custom_read_image = custom_read_image

    def _get_image_path(self, filename):
        matches = self._data.loc[self._data["filename"] == filename]

        if len(matches) == 0:
            raise FileNotFoundError(f"{filename} not found")

        subfolder = "train" if list(matches["in_train"])[0] else "test"
        basefolder = os.path.join(self._folder, subfolder)
        filepath = os.path.join(basefolder, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")

        return filepath

    def _read_image(self, filepath):
        img = cv.imread(filepath)

        if self._image_size:
            return cv.resize(img, self._image_size)

        return img

    def get_image_by_index(self, index):
        filepath = self._get_image_path(self._data["filename"][index])

        if self._custom_read_image is not None:
            self._custom_read_image(filepath)

        return self._read_image(filepath)

    def get_image_by_filename(self, filename):
        filepath = self._get_image_path(filename)

        if self._custom_read_image is not None:
            self._custom_read_image(filepath)

        return self._read_image(filepath)

    def get_image_filename(self, index):
        return self._data["filename"][index]

    def images(self):
        N = len(self._data)
        for i in range(N):
            yield self.get_image_by_index(i)

    def length(self):
        return len(self._data)


if __name__ == "__main__":
    from utils import DATASET_FOLDER

    df = pd.read_pickle("./data/data_info.pkl")
    ds = Dataset(df, DATASET_FOLDER)

    print(ds.length())
    print(ds.get_image_by_index(96382))
    print(ds.get_image_by_filename("2.jpg"))

    # for (img, idx) in ds.images():
    #    print(img)

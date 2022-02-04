import os
import glob
import cv2 as cv
import pandas as pd
from utils import STANDARD_FEATURES_SIZE


class Dataset:
    def __init__(
        self,
        folder,
        image_size=STANDARD_FEATURES_SIZE,
        custom_read_image=None,
        testonly=False,
    ):
        self._folder = folder
        self._testonly = testonly
        self._data = pd.read_pickle(os.path.join(folder, "data_info.pkl"))

        if self._testonly:
            self._data = self._data.loc[~self._data["in_train"]]
            self._prev_test_index = self._data.loc[~self._data["in_train"]].index
            self._data.reset_index(drop=True, inplace=True)

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

    def get_image_index(self, filename):
        for index in range(self._data.shape[0]):  # n_row
            if self._data["filename"][index] == filename:
                return index
        return -1

    def images(self):
        N = len(self._data)
        for i in range(N):
            yield self.get_image_by_index(i)

    def length(self):
        return len(self._data)

    def get_relevant_indexes(self, index):
        """
        List of index with same genre of the 'index' image.
        """
        query_genre = self.get_image_genre_by_index(index)
        docs_genres = self._data.loc[~self._data["in_train"]]["genre"]

        relevant_ids = docs_genres.loc[docs_genres == query_genre].index
        return list(relevant_ids)

    def get_image_genre_by_index(self, index):
        return self._data["genre"][index]

    def get_image_genre_by_filename(self, filename):
        index = self.get_image_index(filename)
        return self.get_image_genre_by_index(index)

    def get_test_indexes(self):
        return list(self._prev_test_index)


if __name__ == "__main__":
    from utils import DATASET_FOLDER

    ds = Dataset(DATASET_FOLDER)

    print(ds.length())
    print(ds.get_image_by_index(0))
    # print(ds.get_image_by_filename("20.jpg"))
    print(ds.get_relevant_indexes(3))

    # for (img, idx) in ds.images():
    #    print(img)

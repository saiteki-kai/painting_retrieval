import os
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from dataset import Dataset
from descriptor import ImageDescriptor
from utils import load_features, FEATURES_FOLDER, OUTPUT_FOLDER, TEST_FOLDER


class ImageRetrieval:
    def __init__(self, feature):
        self.feature = feature

    def search(self, query, n_results=5):
        q = ImageDescriptor.describe(query, self.feature)  # query representation

        with open(os.path.join(FEATURES_FOLDER, f"{self.feature}.index"), 'rb') as index:
            NN = pickle.load(index)
            distances, indexes = NN.kneighbors([q], n_results)

            return dict(zip(indexes[0], distances[0]))

    def index(self, similarity):
        features = load_features(self.feature)
        NN = NearestNeighbors(metric=similarity).fit(features)

        with open(os.path.join(FEATURES_FOLDER, f"{self.feature}.index"), "wb") as index:
            pickle.dump(NN, index)

    def plot_similar_results(self, query, distances, n_results=5, save=False):
        fig, axes = plt.subplots(2, n_results)

        # hide axis
        for ax in axes.ravel():
            ax.set_axis_off()

        # query image
        axes[0, 2].imshow(cv.cvtColor(query, cv.COLOR_BGR2RGB))
        axes[0, 2].set_title("query")

        # ranked similar image
        for n, (idx, dist) in enumerate(distances.items()):
            axes[1, n].imshow(cv.cvtColor(ds.get_image_by_index(idx), cv.COLOR_BGR2RGB))
            axes[1, n].set_title(f"{dist:.2f}")

        if save:
            fig.savefig(
                os.path.join(OUTPUT_FOLDER, f"out_{distances.keys()[0]}_{self.feature}.png"))
        else:
            plt.show()


if __name__ == "__main__":
    ds = Dataset(TEST_FOLDER)

    image = ds.get_image_by_index(50)

    ir = ImageRetrieval("rgb_hist")
    ir.index("euclidean")

    dists = ir.search(image, 5)

    ir.plot_similar_results(image, dists, n_results=5, save=False)

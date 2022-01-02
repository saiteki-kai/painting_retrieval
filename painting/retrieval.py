import os
import pickle
from time import perf_counter

import cv2 as cv
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from dataset import Dataset
from descriptor import describe
from painting.utils import TEST_FOLDER
from utils import load_features, FEATURES_FOLDER, OUTPUT_FOLDER


class ImageRetrieval:
    def __init__(self, feature):
        self.feature = feature

    def search(self, query, similarity="euclidean", n_results=5):
        start_time = perf_counter()

        q = describe(query, self.feature)  # query representation

        filename = f"{self.feature}-{similarity}.index"
        with open(os.path.join(FEATURES_FOLDER, filename), 'rb') as index:
            NN = pickle.load(index)
            distances, indexes = NN.kneighbors([q], n_results)

            elapsed = perf_counter() - start_time
            return indexes[0], distances[0], elapsed

    def search_without_index(self, query, similarity="euclidean", n_results=5):
        start_time = perf_counter()

        # query representation
        q = describe(query, self.feature)

        # load the raw document representation
        features = load_features(self.feature)

        # compute distances between the query and the documents
        distances = pairwise_distances([q], features, metric=similarity)
        distances = distances[0]

        # rank indexes
        ranked_indexes = numpy.argsort(distances)

        ranked_indexes = ranked_indexes[:n_results]
        distances = distances[ranked_indexes]

        elapsed = perf_counter() - start_time
        return ranked_indexes, distances, elapsed

    def index(self, similarity):
        features = load_features(self.feature)
        NN = NearestNeighbors(metric=similarity).fit(features)

        filename = f"{self.feature}-{similarity}.index"
        with open(os.path.join(FEATURES_FOLDER, filename), "wb") as index:
            pickle.dump(NN, index)

    def plot_similar_results(self, query, indexes, distances=None, n_results=5, save=False):
        fig, axes = plt.subplots(2, n_results)

        # hide axis
        for ax in axes.ravel():
            ax.set_axis_off()

        # query image
        axes[0, 2].imshow(cv.cvtColor(query, cv.COLOR_BGR2RGB))
        axes[0, 2].set_title("query")

        # ranked similar image
        for n, idx in enumerate(indexes):
            axes[1, n].imshow(cv.cvtColor(ds.get_image_by_index(idx), cv.COLOR_BGR2RGB))

            if distances is not None:
                axes[1, n].set_title(f"{distances[n]:.2f}")

        if save:
            fig.savefig(
                os.path.join(OUTPUT_FOLDER, f"out_{self.feature}.png"))
        else:
            plt.show()


if __name__ == "__main__":
    ds = Dataset(TEST_FOLDER, (512, 512))

    image = ds.get_image_by_index(410)
    # image = ds.get_image_by_filename("49823.jpg")
    # image = ds.get_image_by_filename("2993.jpg")

    metric = "euclidean"
    results = 5

    ir = ImageRetrieval("rgb_hist")
    ir.index(metric)

    ids, dists, time = ir.search(image, metric, results)
    print(dict(zip(ids, dists)))
    print("time: ", time)

    ids, dists, time = ir.search_without_index(image, metric, results)
    print(dict(zip(ids, dists)))
    print("time: ", time)

    ir.plot_similar_results(image, ids, distances=dists, n_results=results, save=False)

    # histogram: minkowski distance [p=1 (city-block), p=2 (euclidean)]

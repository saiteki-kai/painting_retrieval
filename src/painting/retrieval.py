import os
import pickle
from time import perf_counter

import cv2 as cv
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ..painting.dataset import Dataset
from ..painting.descriptor import compute_feature
from ..painting.evalutation_metrics import *
from ..painting.utils import (
    DATASET_FOLDER,
    FEATURES_FOLDER,
    OUTPUT_FOLDER,
    STANDARD_FEATURES_SIZE,
    load_features,
)


class ImageRetrieval:
    def __init__(self, feature, dataset, evalutation=False):
        self.feature = feature
        self.dataset = dataset
        self.evalutation = evalutation

    def search(self, query, similarity="euclidean", n_results=5):
        start_time = perf_counter()

        # query representation
        if query is int:
            query_img = self.dataset.get_image_by_index(query)
        else:
            query_img = query

        if self.feature == "vgg":
            from features_extraction import get_vgg

            q = get_vgg(image=query_img, cut_level=3)
        elif self.feature == "resnet50":
            from features_extraction import get_resnet50

            q = get_resnet50(image=query_img)
        else:
            query_img = cv.resize(query_img, STANDARD_FEATURES_SIZE)
            q = compute_feature(query_img, self.feature)

        filename = f"{self.feature}-{similarity}.index"

        with open(os.path.join(FEATURES_FOLDER, filename), "rb") as index:
            NN = pickle.load(index)
            distances, indexes = NN.kneighbors([q], n_results)

            distances = distances[0]
            indexes = indexes[0]

            elapsed = perf_counter() - start_time
            return indexes, distances, elapsed

    def search_without_index(self, query_id, similarity="euclidean", n_results=5):
        start_time = perf_counter()

        # query representation
        query_img = self.dataset.get_image_by_index(query_id)

        if self.feature == "vgg":
            from features_extraction import get_vgg

            q = get_vgg(image=query_img, cut_level=3)
        elif self.feature == "resnet50":
            from features_extraction import get_resnet50

            q = get_resnet50(image=query_img)
        else:
            query_img = cv.resize(query_img, STANDARD_FEATURES_SIZE)
            q = compute_feature(query_img, self.feature)

        # load the raw document representation
        features = load_features(self.feature)

        # select only features from the test collection
        if self.evalutation:
            indexes = self.dataset.get_test_indexes()
            features = features[indexes]

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

        # select only features from the test collection
        if self.evalutation:
            indexes = self.dataset.get_test_indexes()
            features = features[indexes]

        NN = NearestNeighbors(metric=similarity).fit(features)

        subfolder = "evaluation" if self.evalutation else ""

        filename = f"{self.feature}-{similarity}.index"
        with open(os.path.join(FEATURES_FOLDER, subfolder, filename), "wb") as index:
            pickle.dump(NN, index)

    def plot_similar_results(
        self, query_idx, doc_indexes, distances=None, n_results=5, save=False
    ):
        fig, axes = plt.subplots(2, n_results)

        # hide axis
        for ax in axes.ravel():
            ax.set_axis_off()

        # query image
        axes[0, 2].imshow(
            cv.cvtColor(self.dataset.get_image_by_index(query_idx), cv.COLOR_BGR2RGB)
        )
        axes[0, 2].set_title("query")

        # ranked similar image
        for n, doc_idx in enumerate(doc_indexes):
            axes[1, n].imshow(
                cv.cvtColor(self.dataset.get_image_by_index(doc_idx), cv.COLOR_BGR2RGB)
            )

            if distances is not None:
                axes[1, n].set_title(f'dist : {distances[n]:.2f}\n"{doc_idx}"')

        if save:
            fig.savefig(os.path.join(OUTPUT_FOLDER, f"out_{self.feature}.png"))
        else:
            plt.show()

    def evaluate_query(
        self, query_id, relevant_ids, metrics, similarity="euclidean", n_results=5
    ):
        retrieved_ids, _, _ = self.search(query_id, similarity, n_results)

        results = {}
        for m in metrics:
            try:
                fn = getattr(evalutation_metrics, m)
                results[m] = fn(relevant_ids, retrieved_ids, k=n_results)
            except Exception:
                raise ValueError(f"Unknown metric function: {m}")

        return results

    def evaluate_queries(
        self, query_ids, relevant_ids, metrics, similarity="euclidean", n_results=5
    ):
        results = []

        for i, query_id in enumerate(query_ids):
            res = self.evaluate_query(
                query_id, relevant_ids[i], metrics, similarity, n_results
            )
            results.append(res)

        return results


def retrieve_images(img, feature, similarity="euclidean", n_features=5):
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    # MATCHING
    # yield primo messaggio nel bot (matching / non matching), se possibile ?

    ir = ImageRetrieval(feature, ds)

    ids, dists, time = ir.search(img, similarity, n_features)
    print("Search Time with index: ", time)

    return [ds.get_image_filepath(idx) for idx in ids], time, dists


if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    query_id = 1320

    feature = "rgb_hist"
    metric = "euclidean"
    results = 5

    ir = ImageRetrieval(feature, ds)
    ir.index(metric)

    ids, dists, time = ir.search_without_index(query_id, metric, results)
    print("Search Time without_index: ", time)
    # print(dict(zip(ids, dists)))

    ids, dists, time = ir.search(query_id, metric, results)
    print("Search Time with index: ", time)
    # print(dict(zip(ids, dists)))

    ir.plot_similar_results(
        query_id, ids, distances=dists, n_results=results, save=False
    )

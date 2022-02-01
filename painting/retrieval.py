import os
import pickle
from time import perf_counter

import cv2 as cv
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import metrics as my_metrics
from dataset import Dataset
from descriptor import compute_feature
from utils import (DATASET_FOLDER, FEATURES_FOLDER, OUTPUT_FOLDER,
                   STANDARD_FEATURES_SIZE, load_features)
from vgg_features_extraction import get_resnet50, get_vgg


class ImageRetrieval:
    def __init__(self, feature, dataset):
        self.feature = feature
        self.dataset = dataset

    def search(self, query_id, similarity="euclidean", n_results=5):
        start_time = perf_counter()

        # query representation
        # VGG want a cut_level
        if self.feature == "vgg":
            q = get_vgg(image=query, cut_level=3)
        elif self.feature == "resnet50":
            q = get_resnet50(image=query)
        else:
            query_img = self.dataset.get_image_by_index(query_id)
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
        # VGG want a cut_level
        if self.feature == "vgg":
            q = get_vgg(image=query, cut_level=3)
        elif self.feature == "resnet50":
            q = get_resnet50(image=query)
        else:
            query_img = self.dataset.get_image_by_index(query_id)
            query_img = cv.resize(query_img, STANDARD_FEATURES_SIZE)
            q = compute_feature(query_img, self.feature)

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
        self, query_id, relevant_ids, metrics, distance="euclidean", n_results=5
    ):
        retrieved_ids, _, _ = self.search(query_id, distance, n_results)

        results = {}
        for m in metrics:
            try:
                fn = getattr(my_metrics, m)
                results[m] = fn(relevant_ids, retrieved_ids, k=n_results)
            except Exception:
                raise ValueError(f"Unknown metric function: {m}")

        return results

    def evaluate_queries(self, query_ids, relevant_ids, metrics, n_results=5):
        results = []

        for i, query_id in enumerate(query_ids):
            res = self.evaluate_query(query_id, relevant_ids[i], metrics, n_results)
            results.append(res)

        return results


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

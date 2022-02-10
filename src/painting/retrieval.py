import os
import pickle
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from src.config import DATASET_FOLDER, FEATURES_FOLDER, OUTPUT_FOLDER, STANDARD_FEATURES_SIZE
from src.config import LIST_GENRE, LIST_F1_PER_GENRE_ON_TEST
from src.painting import evalutation_metrics
from src.painting.dataset import Dataset
from src.painting.descriptor import compute_feature
from src.painting.features_extraction import get_resnet50, prediction_resnet50
from src.painting.utils import load_features


class ImageRetrieval:
    def __init__(self, feature, dataset, evaluation=False):
        self._feature = feature
        self._dataset = dataset
        self._evaluation = evaluation

    def search(self, query, similarity="euclidean", n_results=5):
        start_time = time.time()

        # query representation
        if isinstance(query, int):
            query_img = self._dataset.get_image_by_index(query)
        else:
            query_img = query

        if self._feature == "resnet50":
            q = get_resnet50(image=query_img)
        else:
            q = compute_feature(query_img, self._feature)
            q = q.ravel()

        subfolder = "evaluation" if self._evaluation else ""

        filename = f"{self._feature}-{similarity}.index"

        with open(os.path.join(FEATURES_FOLDER, subfolder, filename), "rb") as index:
            NN = pickle.load(index)
            distances, indexes = NN.kneighbors([q], n_results)

            distances = distances[0]
            indexes = indexes[0]

            elapsed = time.time() - start_time
            return indexes, distances, elapsed

    def search_with_classification(self, query, similarity="euclidean", n_results=5, confidence=0.4):
        start_time = time.time()

        # query representation
        if isinstance(query, int):
            query_img = self._dataset.get_image_by_index(query)
        else:
            query_img = query

        genre_pred = prediction_resnet50(query_img)

        if self._feature == "resnet50":
            q = get_resnet50(image=query_img)
        else:
            q = compute_feature(query_img, self._feature)

        filename = f"{self._feature}-{similarity}.index"

        with open(os.path.join(FEATURES_FOLDER, filename), "rb") as index:
            NN = pickle.load(index)
            distances, indexes = NN.kneighbors([q], self._dataset.length())

            distances = distances[0]
            indexes = indexes[0]

            # Select only the indexes of the images of the same genre
            n_genre = numpy.argmax(genre_pred)
            print("Genre: " + LIST_GENRE[n_genre])

            if LIST_F1_PER_GENRE_ON_TEST[n_genre] >= confidence:
                print("---------------WITH CLASSIFICATION---------------")

                index_genre = self._dataset.get_indexes_from_genre(LIST_GENRE[n_genre])

                indexes_genre = []
                distances_genre = []

                for i in range(len(indexes)):
                    if (indexes[i] in index_genre):
                        indexes_genre.append(indexes[i])
                        distances_genre.append(distances[i])

                if n_results is not None:
                    if n_results < len(indexes_genre):
                        indexes_genre = indexes_genre[:n_results]
                        distances_genre = distances_genre[:n_results]

                elapsed = time.time() - start_time
                return indexes_genre, distances_genre, elapsed
            else:
                print("---------------WITHOUT CLASSIFICATION---------------")
                elapsed = time.time() - start_time
                return indexes, distances, elapsed

    def search_without_index(self, query, similarity="euclidean", n_results=5):
        start_time = time.time()

        # query representation
        if isinstance(query, int):
            query_img = self._dataset.get_image_by_index(query)
        else:
            query_img = query

        if self._feature == "resnet50":
            q = get_resnet50(image=query_img)
        else:
            q = compute_feature(query_img, self._feature)
            q = q.ravel()

        # load the raw document representation
        features = load_features(self._feature)

        # select only features from the test collection
        if self._evaluation:
            indexes = self._dataset.get_test_indexes()
            features = features[indexes]

        # compute distances between the query and the documents
        distances = pairwise_distances([q], features, metric=similarity)
        distances = distances[0]

        # rank indexes
        ranked_indexes = numpy.argsort(distances)

        ranked_indexes = ranked_indexes[:n_results]
        distances = distances[ranked_indexes]

        elapsed = time.time() - start_time
        return ranked_indexes, distances, elapsed

    def index(self, similarity):
        features = load_features(self._feature)

        # select only features from the test collection
        if self._evaluation:
            indexes = self._dataset.get_test_indexes()
            features = features[indexes]

        NN = NearestNeighbors(metric=similarity, algorithm="ball_tree").fit(features)

        subfolder = "evaluation" if self._evaluation else ""

        filename = f"{self._feature}-{similarity}.index"
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
        query_img = self._dataset.get_image_by_index(query_idx)
        query_img = cv.cvtColor(query_img, cv.COLOR_BGR2RGB)
        query_genre = self._dataset.get_image_genre_by_index(query_idx)
        axes[0, 2].imshow(query_img)
        axes[0, 2].set_title(f"query\n {query_genre}")

        # ranked similar image
        for n, doc_idx in enumerate(doc_indexes):
            doc_img = self._dataset.get_image_by_index(doc_idx)
            doc_img = cv.cvtColor(doc_img, cv.COLOR_BGR2RGB)
            axes[1, n].imshow(doc_img)

            if distances is not None:
                is_test = doc_idx in self._dataset.get_test_indexes()
                genre = self._dataset.get_image_genre_by_index(doc_idx)
                axes[1, n].set_title(f'dist : {distances[n]:.2f}\n{genre}\n({"test" if is_test else "all"})')

        if save:
            fig.savefig(os.path.join(OUTPUT_FOLDER, f"out_{self._feature}.png"))
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


def retrieve_images(img, feature, similarity="euclidean", n_results=5):
    image_size = (224, 224) if feature == "resnet50" else STANDARD_FEATURES_SIZE

    ds = Dataset(DATASET_FOLDER, image_size=image_size)
    ir = ImageRetrieval(feature, ds)

    img = cv.resize(img, img_size)

    ids, dists, secs = ir.search(img, similarity, n_results)
    print("Search Time with index: ", secs)

    # ids, dists, secs = ir.search_with_classification(img, similarity)
    # print("Search Time with index: ", secs)

    return [ds.get_image_filepath(idx) for idx in ids], secs, dists

import os
import pickle
from time import perf_counter

import cv2 as cv
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import metrics as my_metrics
from dataset import Dataset
from descriptor import compute_feature
from utils import TRAIN_FOLDER, TEST_FOLDER, RETRIEVAL_FOLDER, FEATURES_FOLDER 
from utils import load_features, FEATURES_FOLDER, OUTPUT_FOLDER
from utils import STANDARD_FEATURES_SIZE

from vgg_features_extraction import get_vgg


class ImageRetrieval:
    def __init__(self, feature, list_files, dataset):
        self.feature = feature
        self.list_files = list_files
        self.dataset = dataset

    def search(self, query, similarity="euclidean", n_results=5):
        start_time = perf_counter()

       # query representation
        if self.feature == 'vgg':
            q = get_vgg(image_path=query, level=3)
        else:
            query = cv.resize(query, STANDARD_FEATURES_SIZE)
            q = compute_feature(query, self.feature)

        filename = f"{self.feature}-{similarity}.index"

        with open(os.path.join(FEATURES_FOLDER, filename), 'rb') as index:
            NN = pickle.load(index)
            distances, indexes = NN.kneighbors([q], n_results)

            elapsed = perf_counter() - start_time
            return indexes[0], distances[0], elapsed

    def search_without_index(self, query, similarity="euclidean", n_results=5):
        start_time = perf_counter()

        # query representation
        if self.feature == 'vgg':
            q = get_vgg(image_path=query, level=3)
        else:
            query = cv.resize(query, STANDARD_FEATURES_SIZE)
            q = compute_feature(query, self.feature)

        # load the raw document representation
        features = load_features(self.feature, self.list_files)

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
        features = load_features(self.feature, self.list_files)
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
        if self.feature == 'vgg':
            query = cv.imread(query)

        axes[0, 2].imshow(cv.cvtColor(query, cv.COLOR_BGR2RGB))
        axes[0, 2].set_title("query")

        # ranked similar image
        for n, idx in enumerate(indexes):
            axes[1, n].imshow(cv.cvtColor(self.dataset.get_image_by_index(idx), cv.COLOR_BGR2RGB))

            if distances is not None:
                name_file = self.dataset._image_list[idx][self.dataset._image_list[idx].rfind('/')+1:]
                #ds._image_list[i][ds._image_list[i].rfind('/')+1:]
                axes[1, n].set_title(f"dist : {distances[n]:.2f}\n\"{name_file}\"")

        if save:
            fig.savefig(
                os.path.join(OUTPUT_FOLDER, f"out_{self.feature}.png"))
        else:
            plt.show()

    def evaluate_query(self, query, true_relevant_ids, metrics, n_results=5):
        ids, _, _ = self.search(query, "minkowski", n_results)

        results = {}
        for m in metrics:
            try:
                fn = getattr(my_metrics, m)
                results[m] = fn(true_relevant_ids, ids, k=n_results)
            except Exception:
                raise ValueError(f"Unknown metric function: {m}")

        return results

    def evaluate_queries(self, queries, true_relevant_ids, metrics, n_results=5):
        results = []

        for i, q in enumerate(queries):
            res = self.evaluate_query(q, true_relevant_ids[i], metrics, n_results)
            results.append(res)

        return results


if __name__ == "__main__":
    ds = Dataset(RETRIEVAL_FOLDER)
    
    #image = ds.get_image_by_filename("34463.jpg")
    index_im = 139
    image = ds.get_image_by_index(index_im)

    feature = "hsv_hist"
    metric = "euclidean"
    results = 5
    list_files = []
    #remove the path, we just want the file names
    for i in range(len(ds._image_list)):
        list_files.append( ds._image_list[i][ds._image_list[i].rfind('/')+1:] )

    
    ir = ImageRetrieval(feature, list_files, ds)
    ir.index(metric)


    if feature == "vgg":
        image = ds._image_list[index_im]

    
    ids, dists, time = ir.search(image, metric, results)
    #print(dict(zip(ids, dists)))
    print("Search Time with index: ", time)
    
    ids, dists, time = ir.search_without_index(image, metric, results)
    #print(dict(zip(ids, dists)))
    print("Search Time without_index: ", time)
    
    #print(dists)
    #print(ids)

    ir.plot_similar_results(image, ids, distances=dists, n_results=results, save=False)

    # histogram: minkowski distance [p=1 (city-block), p=2 (euclidean)]
    from metrics import precision, recall, average_precision
    print("Precision: " + str( precision(ids, n_query=1, type_feature='Color') ))
    print("Recall: " + str( recall(ids, n_query=1, type_feature='Color') ))
    
    """
        #Just a test
        ids = [139, 91, 135, 91, 26] # [R notR R notR R]
        print("Average Precision: " + str( average_precision(ids, n_query=1, type_feature='Color', k=5) ))
        print("Precision: " + str( precision(ids, n_query=1, type_feature='Color') ))
    """

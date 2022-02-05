import os

import cv2 as cv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load

from ..painting.dataset import Dataset
from ..painting.descriptor import compute_feature
from ..painting.utils import (DATASET_FOLDER, FEATURES_FOLDER,
                              LIST_OF_FEATURES_IMPLEMENTED,
                              STANDARD_FEATURES_SIZE)


def compute_descriptor(dataset: Dataset, descriptor_name):
    """
    :param dataset: Dataset instance
    :param descriptor_name: the feature to compute for each image
    """

    if descriptor_name == 'resnet50':
        compute_feature(dataset, descriptor_name)
        return

    N = dataset.length()

    # compute the feature on a random image to get the length
    rand_img = dataset.get_image_by_index(0)
    F_length = len(compute_feature(rand_img, descriptor_name))

    memmap_filepath = os.path.join(FEATURES_FOLDER, f"tmp.memmap")
    descriptor_filepath = os.path.join(FEATURES_FOLDER, f"{descriptor_name}.npy")

    F = np.memmap(memmap_filepath, dtype="float32", mode="w+", shape=(N, F_length))

    dump(F, descriptor_filepath)
    F = load(descriptor_filepath, mmap_mode="r+")

    def fill_matrix(img, idx, F):
        F[idx] = compute_feature(img, descriptor_name)
        # print("[Worker %d] Shape for image %d is %s" % (os.getpid(), idx, F[idx].shape))

    Parallel(n_jobs=-1, verbose=1)(
        delayed(fill_matrix)(img, idx, F) for idx, img in enumerate(dataset.images())
    )


if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    # We avoit to do ccv for now (too slow)
    avoid_list = ['ccv', 'resnet50']
    list_of_features = [x for x in LIST_OF_FEATURES_IMPLEMENTED if (x not in avoid_list )]

    for feature in list_of_features:
        print("Computing: " + feature)
        compute_descriptor(ds, feature)

    # We want to comupute resnet now to observ better the results.
    ds = Dataset(DATASET_FOLDER, (224, 224))

    print("Computing: resnet50")
    compute_descriptor(ds, "resnet50")

import os

import numpy as np
from joblib import Parallel, delayed, dump, load

from src.painting.dataset import Dataset
from src.painting.descriptor import compute_feature
from src.painting.utils import FEATURES_FOLDER


def compute_descriptor(dataset: Dataset, descriptor_name):
    """
    :param dataset: Dataset instance
    :param descriptor_name: the feature to compute for each image
    """

    N = dataset.length()

    if descriptor_name == "resnet50":
        F_length = 1024
    else:
        # compute the feature on a random image to get the length
        rand_img = dataset.get_image_by_index(0)
        feature = compute_feature(rand_img, descriptor_name)

        F_length = len(feature)

    memmap_filepath = os.path.join(FEATURES_FOLDER, "tmp.memmap")
    descriptor_filepath = os.path.join(FEATURES_FOLDER, f"{descriptor_name}.npy")

    F = np.memmap(memmap_filepath, dtype="float32", mode="w+", shape=(N, F_length))

    dump(F, descriptor_filepath)
    F = load(descriptor_filepath, mmap_mode="r+")

    N_JOBS = 8

    if descriptor_name == "resnet50":

        def fill_matrix(pred, idx, F):
            F[idx] = pred

        Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(fill_matrix)(pred, idx, F)
            for idx, pred in enumerate(compute_feature(dataset, "resnet50"))
        )
    else:

        def fill_matrix(img, idx, F):
            F[idx] = compute_feature(img, descriptor_name)

        Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(fill_matrix)(img, idx, F)
            for idx, img in enumerate(dataset.images())
        )

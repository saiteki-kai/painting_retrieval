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

    if descriptor_name == 'resnet50':
        compute_feature(dataset, descriptor_name)
        return

    # if descriptor_name == 'orb':
    #     compute_feature(dataset, descriptor_name)
    #     return

    N = dataset.length()

    # compute the feature on a random image to get the length
    rand_img = dataset.get_image_by_index(0)
    feature = compute_feature(rand_img, descriptor_name)

    if descriptor_name == "orb":
        feature = feature[0]

    F_length = len(feature)

    memmap_filepath = os.path.join(FEATURES_FOLDER, "tmp.memmap")
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

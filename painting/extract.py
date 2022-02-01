import os

import numpy as np
import pandas as pd
import cv2 as cv
from joblib import Parallel, delayed, load, dump

from dataset import Dataset
from descriptor import compute_feature
from utils import DATASET_FOLDER, FEATURES_FOLDER
from utils import STANDARD_FEATURES_SIZE
from utils import LIST_OF_FEATURES_IMPLEMENTED


def compute_descriptor(dataset: Dataset, descriptor_name, vgg_level=1):
    """
    :param dataset: Dataset instance
    :param descriptor_name: the feature to compute for each image
    """
    if descriptor_name == "vgg":
        compute_feature(dataset, descriptor_name, vgg_level=vgg_level)
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
    df = pd.read_pickle("./data/data_info.pkl")
    ds = Dataset(df, DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    # We avoit to do ccv for now (too slow)
    list_of_features = [
        x for x in LIST_OF_FEATURES_IMPLEMENTED if (x != "ccv" and x != "vgg")
    ]
    # list_of_features = ["rgb_hist", "hsv_hist", "lbp"]

    for feature in list_of_features:
        print("Computing: " + feature)
        compute_descriptor(ds, feature)

    # VGG features are special for now, we want to be able to specify the level
    # Once a level is fixed we can modify it and it will be the same of
    # the others

    ds_vgg = Dataset(DATASET_FOLDER, (224, 224))
    print("Computing: vgg")
    compute_descriptor(ds_vgg, "vgg", vgg_level=3)

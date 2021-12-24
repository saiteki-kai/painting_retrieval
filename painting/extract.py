import os

import numpy as np

from painting.dataset import Dataset
from painting.descriptor import compute_rgb_hist, compute_hsv_hist, compute_lbp
from painting.utils import TRAIN_FOLDER, FEATURES_FOLDER


def compute_descriptor(dataset: Dataset, feature_size, descriptor_fn):
    """
    :param dataset: Dataset instance
    :param feature_size: features vector length
    :param descriptor_fn: function that computes the feature for each image
    :return features matrix
    """

    if not isinstance(feature_size, int):
        raise ValueError("'features_size' must be an integer number")

    if not callable(descriptor_fn):
        raise ValueError("'descriptor_fn' must be callable")

    N = dataset.length()
    features = np.zeros((N, feature_size))
    for idx, img in enumerate(dataset.images()):
        f = descriptor_fn(img)
        features[idx, :] = f
        # print(f"{img.shape} -> {f.shape}")
        if idx % 1000 == 0:
            print(idx)
    return features


if __name__ == "__main__":
    ds = Dataset(TRAIN_FOLDER, (512, 512))

    #features = compute_descriptor(ds, 692, compute_hsv_hist)
    #np.save(os.path.join(FEATURES_FOLDER, "hsv_hist"), features)

    # features = compute_descriptor(ds, 768, compute_rgb_hist)
    # np.save(os.path.join(FEATURES_FOLDER, "rgb_hist"), features)

    features = compute_descriptor(ds, 18, compute_lbp)
    np.save(os.path.join(FEATURES_FOLDER, "lbp"), features)

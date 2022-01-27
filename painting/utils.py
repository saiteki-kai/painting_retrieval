import os.path

import numpy as np

TRAIN_FOLDER = os.path.join(os.getcwd(), "data", "raw", "train")
TEST_FOLDER = os.path.join(os.getcwd(), "data", "raw", "test")
RETRIEVAL_FOLDER = os.path.join(os.getcwd(), "data", "raw", "retrieval", "data")
FEATURES_FOLDER = os.path.join(os.getcwd(), "data", "features")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "out")

STANDARD_FEATURES_SIZE = (512, 512)


def load_features(feature, list_files):
    list_of_features = [
        "hog", "hsv_hist", "lbp", 
        "rgb_hist", "bow_sift",
        "ccv", "dct"]
        
    if feature not in list_of_features:
        raise ValueError(f"unrecognized feature: '{feature}'")

    filepath = os.path.join(FEATURES_FOLDER, feature)

    if not os.path.exists(filepath):
        raise IOError(f"feature '{feature}' not saved")

    N = len(list_files)
    #This is just to set the feature_size)
    f = np.load( os.path.join(filepath, list_files[0]+".npy") )
    feature_size = len( f )
    saved_features = np.zeros((N, feature_size))
    saved_features[0, :] = f

    for i in range(1, N):
        f = np.load( os.path.join(filepath, list_files[i]+".npy") )
        saved_features[i, :] = f

    return saved_features

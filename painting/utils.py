import os
import json
import numpy as np

DATA_FOLDER = os.path.join(os.getcwd(), "data")
TRAIN_FOLDER = os.path.join(os.getcwd(), "data", "raw", "train")
TEST_FOLDER = os.path.join(os.getcwd(), "data", "raw", "test")
RETRIEVAL_FOLDER = os.path.join(os.getcwd(), "data", "raw", "retrieval", "data")
FEATURES_FOLDER = os.path.join(os.getcwd(), "data", "features")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "out")

STANDARD_FEATURES_SIZE = (512, 512)

# For semplicity let ccv the last (not mandatory)
LIST_OF_FEATURES_IMPLEMENTED = [
        "rgb_hist", 
        "hsv_hist",
        "lbp",
        "hog",
        "dct",
        "vgg",
        "ccv"]


def load_features(feature, list_files):
        
    if feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    filepath = os.path.join(FEATURES_FOLDER, feature)

    if not os.path.exists(filepath):
        raise IOError(f"feature '{feature}' not saved")

    N = len(list_files)
    f = np.load( os.path.join(filepath, list_files[0]+".npy") )
    #Just for performance 
    feature_size = len( f )
    saved_features = np.zeros((N, feature_size))
    saved_features[0, :] = f

    for i in range(1, N):
        f = np.load( os.path.join(filepath, list_files[i]+".npy") )
        saved_features[i, :] = f

    return saved_features


def load_groundtruth(query_id=None, feature_type=None):
    with open(os.path.join(DATA_FOLDER, 'groundtruth.json')) as fp:
        qrels = json.load(fp)

        if query_id is None:
            return qrels

        q_obj = qrels[query_id]
        if q_obj and feature_type is not None:
            return q_obj['feature_types'][feature_type]
    
    return None

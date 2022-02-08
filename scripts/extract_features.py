import cv2 as cv

from src.config import DATASET_FOLDER, LIST_OF_FEATURES_IMPLEMENTED, STANDARD_FEATURES_SIZE
from src.painting.dataset import Dataset
from src.painting.extract import compute_descriptor
from src.painting.exact_matching import compute_orb_descriptor
from src.painting.utils import resize_with_max_ratio


def read_image_with_max_ratio(filepath):
    img = cv.imread(filepath)
    img = resize_with_max_ratio(img, 512, 512)
    return img


if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    # We avoid computing ccv for now (too slow)
    avoid_list = ['ccv', 'resnet50']
    list_of_features = [x for x in LIST_OF_FEATURES_IMPLEMENTED if x not in avoid_list]

    # list_of_features = ['local_rgb_hist', 'rgb_hist', 'lbp']
    for feature in list_of_features:
        print("Computing: " + feature)
        compute_descriptor(ds, feature)

    # We want to compute resnet now to observe better the results.
    ds = Dataset(DATASET_FOLDER, (224, 224))

    print("Computing: resnet50")
    compute_descriptor(ds, "resnet50")

    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    print("Computing: ORB")
    compute_orb_descriptor(ds)

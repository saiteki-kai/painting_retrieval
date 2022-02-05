from src.config import DATASET_FOLDER, LIST_OF_FEATURES_IMPLEMENTED, STANDARD_FEATURES_SIZE
from src.painting.dataset import Dataset
from src.painting.extract import compute_descriptor

if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    # We avoid computing ccv for now (too slow)
    avoid_list = ['ccv', 'resnet50']
    list_of_features = [x for x in LIST_OF_FEATURES_IMPLEMENTED if x not in avoid_list]

    # list_of_features = ["rgb_hist"]
    for feature in list_of_features:
        print("Computing: " + feature)
        compute_descriptor(ds, feature)

    # We want to compute resnet now to observ better the results.
    ds = Dataset(DATASET_FOLDER, (224, 224))

    print("Computing: resnet50")
    compute_descriptor(ds, "resnet50")

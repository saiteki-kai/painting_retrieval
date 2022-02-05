import os

from src.config import DATASET_FOLDER, LIST_OF_FEATURES_IMPLEMENTED, FEATURES_FOLDER, \
    SIMILARITY_DISTANCES, STANDARD_FEATURES_SIZE
from src.painting.dataset import Dataset
from src.painting.retrieval import ImageRetrieval

if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    for feature in LIST_OF_FEATURES_IMPLEMENTED:
        # check if already computed
        feature_path = os.path.join(FEATURES_FOLDER, f"{feature}.npy")

        print(f"computing {feature}")

        if os.path.exists(feature_path):
            ir = ImageRetrieval(feature, ds)

            for similarity in SIMILARITY_DISTANCES:
                ir.index(similarity)
        else:
            print(f"{feature} not computed. skip")

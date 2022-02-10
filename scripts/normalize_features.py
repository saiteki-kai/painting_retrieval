import os

from src.config import DATASET_FOLDER, LIST_OF_FEATURES_IMPLEMENTED, FEATURES_FOLDER, STANDARD_FEATURES_SIZE
from src.painting.dataset import Dataset
from src.painting.utils import load_features
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    for feature in LIST_OF_FEATURES_IMPLEMENTED:
        # check if already computed
        feature_path = os.path.join(FEATURES_FOLDER, f"{feature}.npy")

        if os.path.exists(feature_path):
            F = load_features(feature)

            scaler = MinMaxScaler()
            scaler.fit(F)

            dump(scaler, os.path.join(FEATURES_FOLDER, f"scaler_{feature}"))

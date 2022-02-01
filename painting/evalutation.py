from dataset import Dataset
from metrics import recall_at_k, precision_at_k
from retrieval import ImageRetrieval
from utils import DATASET_FOLDER, FEATURES_FOLDER
from utils import STANDARD_FEATURES_SIZE
import pandas as pd

if __name__ == "__main__":
    df = pd.read_pickle("./data/data_info.pkl")
    ds = Dataset(df, DATASET_FOLDER, STANDARD_FEATURES_SIZE)

    ir = ImageRetrieval("rgb_hist", ds)

    query_id = 223

    retrieved_ids, _, _ = ir.search(query_id, "euclidean", n_results=10)

    # TODO
    relevant_ids = list([223, -1, 1467, -1, -1, 1441, 64, 533, 207, 1317])

    res = ir.evaluate_query(
        query_id,
        relevant_ids,
        metrics=["precision_at_k", "recall_at_k"],
        distance="euclidean",
        n_results=10,
    )

    print(res)

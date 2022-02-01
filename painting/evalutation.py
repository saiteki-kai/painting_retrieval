from dataset import Dataset
from metrics import precision_at_k, recall_at_k
from retrieval import ImageRetrieval
from utils import DATASET_FOLDER, FEATURES_FOLDER, STANDARD_FEATURES_SIZE

if __name__ == "__main__":
    ds = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE)

    ir = ImageRetrieval("rgb_hist", ds)

    query_id = 1300

    relevant_ids = ds.get_relevant_indexes(query_id)

    res = ir.evaluate_query(
        query_id,
        relevant_ids,
        metrics=["precision_at_k", "recall_at_k", "average_precision"],
        distance="euclidean",
        n_results=10,
    )

    print(res)

from dataset import Dataset
from evalutation_metrics import precision_at_k, recall_at_k
from retrieval import ImageRetrieval
from utils import DATASET_FOLDER, FEATURES_FOLDER, STANDARD_FEATURES_SIZE

if __name__ == "__main__":
    dataset = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE, testonly=True)

    ir = ImageRetrieval("rgb_hist", dataset, evalutation=True)
    ir.index("euclidean")

    query_id = 1

    relevant_ids = dataset.get_relevant_indexes(query_id)

    res = ir.evaluate_query(
        query_id,
        relevant_ids,
        metrics=["precision_at_k", "recall_at_k", "average_precision"],
        distance="euclidean",
        n_results=5,
    )

    print(res)

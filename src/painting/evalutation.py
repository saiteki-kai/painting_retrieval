from src.painting.dataset import Dataset
from src.painting.retrieval import ImageRetrieval
from src.painting.utils import DATASET_FOLDER, STANDARD_FEATURES_SIZE

if __name__ == "__main__":
    dataset = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE, test_only=True)

    ir = ImageRetrieval("rgb_hist", dataset, evalutation=True)
    ir.index("euclidean")

    query_ids = list(range(1, dataset.length()))
    relevant_ids = [dataset.get_relevant_indexes(query_id) for query_id in query_ids]

    res = ir.evaluate_queries(
        query_ids,
        relevant_ids,
        metrics=["precision_at_k", "recall_at_k", "average_precision"],
        similarity="euclidean",
        n_results=5,
    )

    print(res)

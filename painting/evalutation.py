from painting.dataset import Dataset
from painting.metrics import recall_at_k, precision_at_k
from painting.retrieval import ImageRetrieval
from painting.utils import TEST_FOLDER

if __name__ == "__main__":
    ds = Dataset(TEST_FOLDER, (512, 512))

    retrieval = ImageRetrieval("rgb_hist")
    # retrieval.index("minkowski")

    image = ds.get_image_by_index(410)

    retrieved_ids, _, _ = retrieval.search(image, "minkowski", n_results=10)
    relevant_ids = list([5129, 6309, 6499, -1, 1876, 503, -1, -1, 2844, -1])

    r = recall_at_k(relevant_ids, retrieved_ids)
    p = precision_at_k(relevant_ids, retrieved_ids)
    print(f"Precision: {p:.2f}")
    print(f"Recall: {r:.2f}")
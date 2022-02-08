from src.config import DATASET_FOLDER, STANDARD_FEATURES_SIZE
from src.painting.dataset import Dataset
from src.painting.retrieval import ImageRetrieval

if __name__ == "__main__":
    QUERY_ID = 4445

    FEATURE = "resnet50"
    SIMILARITY = "euclidean"
    RESULTS = 10

    ds = Dataset(DATASET_FOLDER, image_size=(224, 224) if FEATURE == "resnet50" else STANDARD_FEATURES_SIZE)

    ir = ImageRetrieval(FEATURE, ds)
    ir.index(SIMILARITY)

    ids, dists, time = ir.search_without_index(QUERY_ID, SIMILARITY, RESULTS)
    print("Search Time without_index: ", time)
    print(dict(zip(ids, dists)))

    ids, dists, time = ir.search(QUERY_ID, SIMILARITY, RESULTS)
    print("Search Time with index: ", time)
    print(dict(zip(ids, dists)))

    ir.plot_similar_results(
        QUERY_ID, ids, distances=dists, n_results=RESULTS, save=False
    )

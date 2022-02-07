import numpy as np
import pandas as pd

from src.config import DATASET_FOLDER, STANDARD_FEATURES_SIZE
from src.painting.dataset import Dataset
from src.painting.retrieval import ImageRetrieval

if __name__ == "__main__":
    dataset = Dataset(DATASET_FOLDER, image_size=STANDARD_FEATURES_SIZE, test_only=True)

    ir = ImageRetrieval("resnet50", dataset, evaluation=True)
    ir.index("euclidean")

    query_ids = list(range(1, dataset.length()))
    query_genres = [dataset.get_image_genre_by_index(query_id) for query_id in query_ids]
    relevant_ids = [dataset.get_relevant_indexes(query_id) for query_id in query_ids]

    relevant_ids = pd.DataFrame({"genre": query_genres, "query_id": query_ids, "docs_ids": relevant_ids})
    print(relevant_ids)

    metrics = ["precision_at_k", "recall_at_k", "average_precision"]

    for genre, row in relevant_ids.groupby('genre'):
        print(f"Evaluate: {genre}")
        q_ids = list(row['query_id'])
        d_ids = list(row['docs_ids'])

        print(q_ids)

        print(len(q_ids))
        print(len(d_ids))

        results = ir.evaluate_queries(
            q_ids,
            d_ids,
            metrics=metrics,
            similarity="euclidean",
            n_results=5,
        )

        avg_results = {m: [] for m in metrics}

        for res in results:
            for m in metrics:
                avg_results[m].append(res[m])

        avg_results = {key: np.average(avg_results[key]) for key in avg_results.keys()}
        print(avg_results)

import os
import time
import numpy as np
import pandas as pd
import json

from src.config import DATASET_FOLDER, STANDARD_FEATURES_SIZE, OUTPUT_FOLDER
from src.painting.dataset import Dataset
from src.painting.retrieval import ImageRetrieval

if __name__ == "__main__":

    FEATURE = "bow"
    SIMILARITY = "euclidean"
    RESULTS = 10

    image_size = (224, 224) if FEATURE == "resnet50" else STANDARD_FEATURES_SIZE
    dataset = Dataset(DATASET_FOLDER, image_size=image_size, test_only=True)

    ir = ImageRetrieval(FEATURE, dataset, evaluation=True)
    ir.index(SIMILARITY)

    query_ids = list(range(1, dataset.length()))
    query_genres = [dataset.get_image_genre_by_index(query_id) for query_id in query_ids]
    relevant_ids = [dataset.get_relevant_indexes(query_id) for query_id in query_ids]

    relevant_ids = pd.DataFrame({"genre": query_genres, "query_id": query_ids, "docs_ids": relevant_ids})
    relevant_ids = relevant_ids.groupby("genre").sample(45, random_state=1234)

    metrics = ["precision_at_k", "recall_at_k", "average_precision"]

    genre_results = {}
    avg_results = {m: [] for m in metrics}

    for i, (genre, row) in enumerate(relevant_ids.groupby('genre')):
        print(f"[{i + 1:2d}/20] {genre} ({len(row)} paintings)")
        q_ids = list(row['query_id'])
        d_ids = list(row['docs_ids'])

        start = time.time()
        results = ir.evaluate_queries(
            q_ids,
            d_ids,
            metrics=metrics,
            similarity=SIMILARITY,
            n_results=RESULTS,
        )
        print(time.time() - start)

        avg_genre_results = {m: [] for m in metrics}

        for res in results:
            for m in metrics:
                avg_genre_results[m].append(res[m])
                avg_results[m].append(res[m])

        avg_genre_results = {key: np.average(avg_genre_results[key]) for key in avg_genre_results.keys()}
        print(avg_genre_results)

        genre_results[genre] = avg_genre_results

    avg_results = {key: np.average(avg_results[key]) for key in avg_results.keys()}
    print(avg_results)

    res = {"config": {"feature": FEATURE, "similarity": SIMILARITY, "n_results": RESULTS}, "avg": avg_results,
           "by_genre": genre_results}
    json.dump(res, open(os.path.join(OUTPUT_FOLDER, "results", f"{FEATURE}_{SIMILARITY}_@{RESULTS}.json"), "w"))

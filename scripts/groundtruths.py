import os
import glob
import json
import warnings

queries_folder = os.path.join("data", "raw", "retrieval", "data")

qrels = {}
for query_folder in glob.glob(os.path.join(queries_folder, "**")):
    query_id = os.path.split(query_folder)[-1]

    query_image_fp = glob.glob(os.path.join(query_folder, "*.jpg"))

    if len(query_image_fp) == 1:
        query_image_fp = glob.glob(os.path.join(query_folder, "*.jpg"))[0]
    else:
        warnings.warn(
            f"No unique query image found. You must have only one image in the root folder. Skip {query_id}."
        )
        continue

    q_obj = {
        "query_image": os.path.basename(query_image_fp),
        "feature_types": {},
    }

    for feature_type in ["Color", "Global", "Texture"]:
        relevant_folder = os.path.join(query_folder, feature_type, "relevant")

        relevant_docs = glob.glob(os.path.join(relevant_folder, "*.jpg"))
        relevant_docs = [os.path.basename(filepath) for filepath in relevant_docs]

        q_obj["feature_types"][feature_type] = relevant_docs

    qrels[query_id] = q_obj

with open("./data/groundtruth.json", "w") as fp:
    json.dump(qrels, fp)

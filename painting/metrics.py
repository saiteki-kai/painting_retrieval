def precision_at_k(relevant, retrieved, k=None):
    """Precision at k.

    Computes the precision of the first k documents.

    Parameters
    ----------
    relevant: set, list
        A set of indexes of relevant documents.

    retrieved: set, list
        A set of indexes of retrieved documents.

    k: int or None
        The number of documents for the evaluation. If not specified,
        the value is set to the number of documents retrieved.

    Returns
    -------
    float
        The precision of the first k documents.
    """
    if k is None:
        k = len(retrieved)
    elif k <= 0:
        raise ValueError("'k' must be at least 1")

    relevant = set(relevant)
    retrieved = set(retrieved[:k])
    return len(relevant & retrieved) / float(k)


def recall_at_k(relevant, retrieved, k=None):
    """Recall at k.

    Computes the recall of the first k documents.

    Parameters
    ----------
    relevant: set, list
        A set of indexes of relevant documents.

    retrieved: set, list
        A set of indexes of retrieved documents.

    k: int or None
        The number of documents for the evaluation. If not specified,
        the value is set to the number of relevant documents.

    Returns
    -------
    float
        The recall of the first k documents.
    """
    if k is None:
        k = len(relevant)
    elif k <= 0:
        raise ValueError("'k' must be at least 1")

    relevant = set(relevant)
    retrieved = set(retrieved[:k])
    return len(relevant & retrieved) / float(len(relevant))


# FIX: wrong implementation. the average must be computed only using relevant documents
def average_precision(relevant, retrieved, k):
    ap = 0
    for i in range(1, k + 1):
        pi = precision_at_k(relevant, retrieved, i)
        ap = ap + pi
    return ap / min(k, len(relevant))


from utils import RETRIEVAL_FOLDER
from dataset import Dataset
import os
import cv2 as cv

def precision(retrieved_indexes, n_query:int, type_feature:str):

    if type_feature not in ['Color', 'Global', 'Texture']:
        error_string = f"unrecognized feature type: '{type_feature}' "
        error_string += "features type should be 'Color', 'Global' or 'Texture'. "
        raise ValueError(error_string)

    # father of RETRIEVAL_FOLDER
    relevant_path = os.path.abspath(os.path.join(RETRIEVAL_FOLDER, os.pardir))
    relevant_path = os.path.join(relevant_path, str(n_query)+'_query', type_feature, 'relevant')

    dataset = Dataset(RETRIEVAL_FOLDER)
    name_retrieved_files = []
    for idx in retrieved_indexes:
        name_retrieved_files.append(dataset._image_list[idx][dataset._image_list[idx].rfind('/')+1:])

    relevant_dataset = Dataset(relevant_path)
    name_relevant_files = []
    for idx in range(len(relevant_dataset._image_list)):
        name_relevant_files.append(relevant_dataset._image_list[idx][relevant_dataset._image_list[idx].rfind('/')+1:])


    n_retrieved_element = len(retrieved_indexes)
    n_relevant_element_retrieved = 0

    for retrieved in name_retrieved_files:
        if retrieved in name_relevant_files:
            n_relevant_element_retrieved += 1

    return n_relevant_element_retrieved/n_retrieved_element


def recall(retrieved_indexes, n_query:int, type_feature:str):

    if type_feature not in ['Color', 'Global', 'Texture']:
        error_string = f"unrecognized feature type: '{type_feature}' "
        error_string += "features type should be 'Color', 'Global' or 'Texture'. "
        raise ValueError(error_string)

    # father of RETRIEVAL_FOLDER
    relevant_path = os.path.abspath(os.path.join(RETRIEVAL_FOLDER, os.pardir))
    relevant_path = os.path.join(relevant_path, str(n_query)+'_query', type_feature, 'relevant')

    dataset = Dataset(RETRIEVAL_FOLDER)
    name_retrieved_files = []
    for idx in retrieved_indexes:
        name_retrieved_files.append(dataset._image_list[idx][dataset._image_list[idx].rfind('/')+1:])

    relevant_dataset = Dataset(relevant_path)
    name_relevant_files = []
    for idx in range(len(relevant_dataset._image_list)):
        name_relevant_files.append(relevant_dataset._image_list[idx][relevant_dataset._image_list[idx].rfind('/')+1:])
    

    n_relevant_element = len(name_relevant_files)
    n_relevant_element_retrieved = 0

    for retrieved in name_retrieved_files:
        if retrieved in name_relevant_files:
            n_relevant_element_retrieved += 1

    return n_relevant_element_retrieved/n_relevant_element

def precision_at_k(retrieved_indexes, n_query:int, type_feature:str, k:int=None):
    if k is None:
        k = len(retrieved_indexes)
    elif k <= 0:
        raise ValueError("'k' must be at least 1")
    elif k > len(retrieved_indexes): 
        raise ValueError(f"'k' must be smaller than {len(retrieved_indexes)}")

    return precision(retrieved_indexes[:k], n_query, type_feature)
    

def recall_at_k(retrieved_indexes, n_query:int, type_feature:str, k:int=None):
    if k is None:
        k = len(retrieved_indexes)
    elif k <= 0:
        raise ValueError("'k' must be at least 1")
    elif k > len(retrieved_indexes): 
        raise ValueError(f"'k' must be smaller than {len(retrieved_indexes)}")

    return recall(retrieved_indexes[:k], n_query, type_feature)

def average_precision(retrieved_indexes, n_query:int, type_feature:str, k:int=None):
    if k is None:
        k = len(retrieved_indexes)
    elif k <= 0:
        raise ValueError("'k' must be at least 1")
    elif k > len(retrieved_indexes): 
        raise ValueError(f"'k' must be smaller than {len(retrieved_indexes)}")

    # father of RETRIEVAL_FOLDER
    relevant_path = os.path.abspath(os.path.join(RETRIEVAL_FOLDER, os.pardir))
    relevant_path = os.path.join(relevant_path, str(n_query)+'_query', type_feature, 'relevant')

    dataset = Dataset(RETRIEVAL_FOLDER)
    name_retrieved_files = []
    for idx in retrieved_indexes:
        name_retrieved_files.append(dataset._image_list[idx][dataset._image_list[idx].rfind('/')+1:])

    relevant_dataset = Dataset(relevant_path)
    name_relevant_files = []
    for idx in range(len(relevant_dataset._image_list)):
        name_relevant_files.append(relevant_dataset._image_list[idx][relevant_dataset._image_list[idx].rfind('/')+1:])
    

    ap = 0
    relevant_doc = 0
    for ki in range(k):
        if name_retrieved_files[ki] in name_relevant_files:
            #print(retrieved_indexes[ki])
            pi = precision_at_k(retrieved_indexes, n_query, type_feature, ki+1)
            ap = ap + pi

            relevant_doc += 1

    return ap / relevant_doc
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


def average_precision(relevant, retrieved, k=None):
    n = 0  # number of relevant docs
    ap = 0

    for i in range(k):
        if retrieved[i] in relevant:
            pi = precision_at_k(relevant, retrieved, i + 1)
            ap = ap + pi
            n = n + 1

    if n == 0:
        return 0

    return ap / n


if __name__ == "__main__":
    relevant = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    retrieved = [1, 0, 3, 0, 0, 6, 0, 0, 9, 10]

    for i in range(10):
        p = precision_at_k(relevant, retrieved, i + 1)
        print(f"P@{i+1}:", p)

    p1 = precision_at_k(relevant, retrieved, 1)
    p3 = precision_at_k(relevant, retrieved, 3)
    p6 = precision_at_k(relevant, retrieved, 6)
    p9 = precision_at_k(relevant, retrieved, 9)
    p10 = precision_at_k(relevant, retrieved, 10)

    print("AP:", (p1 + p3 + p6 + p9 + p10) / 5)

    print("AP:", average_precision(relevant, retrieved, 10))

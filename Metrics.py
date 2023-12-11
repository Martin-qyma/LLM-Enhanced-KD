import torch
import numpy as np


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    LOG_ARANGE = np.log2(np.arange(20 + 2) + 1e-9)
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1.0 / LOG_ARANGE[2 : k + 2], axis=1)
    dcg = pred_data * (1.0 / LOG_ARANGE[2 : k + 2])
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        r.append(pred)
    return np.array(r, dtype=np.float32)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precision = np.sum(right_pred) / precis_n
    return {"recall": recall, "precision": precision}


def test_one_batch(X):
    # X = zip(rating_list, groundTrue_list)
    sorted_items = X[0]
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []

    ret = RecallPrecision_ATk(groundTrue, r, k=20)
    pre.append(ret["precision"])
    recall.append(ret["recall"])
    ndcg.append(NDCGatK_r(groundTrue, r, k=20))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
    }

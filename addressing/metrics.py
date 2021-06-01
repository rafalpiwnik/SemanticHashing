import heapq

import numpy as np
from tqdm import tqdm

from addressing.MedianHash import medhash_transform, MedianHash


# Precision recall test schema
def run_recall_test(train_pred: np.ndarray, train_targets: np.ndarray,
                    test_pred: np.ndarray, test_targets: np.ndarray, k: int = 100):
    assert train_targets.ndim == 1
    assert test_targets.ndim == 1

    train_codes = medhash_transform(train_pred)
    test_codes = medhash_transform(test_pred)

    precision_scores = []

    for idx, tc in tqdm(enumerate(test_codes)):
        r = precision(test_targets[idx], train_targets, top_k_indices(tc, train_codes, k)[0], k)
        precision_scores.append(r)

    print("Mean precision:")
    print(np.array(precision_scores).mean())
    print(precision_scores)


def precision(actual_target: int, train_targets: np.ndarray, indices: list[int], k: int):
    """Precision is the number of relevant retrieved documents divided by the total number of documents retrieved"""
    prediction_targets: np.ndarray = train_targets[indices]
    hits = np.count_nonzero(prediction_targets == actual_target)
    return hits / float(k)


def top_k_indices(code: MedianHash, pool: list[MedianHash], k: int = 100):
    """Returns indices in pool MedianHash list which correspond to k closest binary codes with respect
    to Hamming distance and all distances of pool MedianHashes to code MedianHash"""
    distances = np.array([code.hamming(p) for p in pool])
    indices = heapq.nsmallest(k, range(len(distances)), key=lambda x: distances[x])
    return indices, distances

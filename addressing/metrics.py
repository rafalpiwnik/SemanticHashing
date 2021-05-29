import heapq

import numpy as np
from tqdm import tqdm

from addressing.MedianHash import medhash_transform, MedianHash


def run_recall_test(train_pred: np.ndarray, train_targets: np.ndarray,
                    test_pred: np.ndarray, test_targets: np.ndarray, k: int = 100):
    assert train_targets.ndim == 1
    assert test_targets.ndim == 1

    train_codes = medhash_transform(train_pred)
    test_codes = medhash_transform(test_pred)

    precision_scores = []

    for idx, tc in tqdm(enumerate(test_codes)):
        r = precision(test_targets[idx], train_targets, top_k_indices(tc, train_codes, k), k)
        precision_scores.append(r)

    # This recall is as expected
    print("Mean precision:")
    print(np.array(precision_scores).mean())
    print(precision_scores)


def precision(actual_target: int, train_targets: np.ndarray, indices: list[int], k: int):
    prediction_targets: np.ndarray = train_targets[indices]
    hits = np.count_nonzero(prediction_targets == actual_target)
    return hits / float(k)


def top_k_indices(code: MedianHash, pool: list[MedianHash], k: int = 100):
    # TODO rework
    distances = np.array([code.hamming(p) for p in pool])
    indices = heapq.nsmallest(k, range(len(distances)), key=lambda x: distances[x])
    return indices

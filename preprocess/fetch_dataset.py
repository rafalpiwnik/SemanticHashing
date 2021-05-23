import os
from typing import Union

import h5py
import h5sparse
import sklearn.utils
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer

AVAILABLE_DATASETS = ["20ng", "rcv1"]


def fetch_dataset(name: str, data_home: Union[str, os.PathLike]) -> sklearn.utils.Bunch:
    """Fetches a dataset to data_home path and returns a Bunch
    Parameters
    ----------
    name : str
        Qualified dataset name
    data_home : str or os.PathLike
        Output dir for fetched data

    Returns
    -------
    sklearn.utils.Bunch
        Dict-like with keys {data, target} among others

    Raises
    -------
    ValueError
        When dataset name is not supported
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError("Invalid dataset name")
    else:
        try:
            os.mkdir(f"{data_home}/{name}")
        except FileExistsError:
            pass

        dest = f"{data_home}/{name}"

        if name == "20ng":
            # TODO progress bar here
            print("Fetching 20ng...")
            train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
            test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

            print("Vectorizing...")
            v = TfidfVectorizer(stop_words="english", max_features=10000)

            # Scipy sparse matrices
            sparse_train_tfidf = v.fit_transform(train.data)
            sparse_test_tfidf = v.transform(test.data)

            with h5sparse.File(f"{dest}/dataSparse.npz", "w") as hf:
                train_docs = hf.create_dataset(name="train", data=sparse_train_tfidf)
                train_labels = hf.create_dataset(name="train_labels", data=train.target)
                test_docs = hf.create_dataset(name="test", data=sparse_test_tfidf)
                test_labels = hf.create_dataset(name="test_labels", data=test.target)

            with h5py.File(f"{dest}/data.hdf5", "w") as hf:
                train_docs = hf.create_dataset(name="train", data=sparse_train_tfidf.toarray(), compression="gzip")
                train_labels = hf.create_dataset(name="train_labels", data=train.target)
                test_docs = hf.create_dataset(name="test", data=sparse_test_tfidf.toarray(), compression="gzip")
                test_labels = hf.create_dataset(name="test_labels", data=test.target)
        elif name == "rcv1":
            return fetch_rcv1()

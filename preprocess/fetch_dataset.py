import json
import os
from typing import Union

import h5py
import numpy as np
import sklearn.utils
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer

from controllers.usersetup import load_config

AVAILABLE_DATASETS = ["20ng", "rcv1"]


def fetch_dataset(name: str, data_home: Union[str, os.PathLike]):
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
            fetch_20ng(10000)
        elif name == "rcv1":
            print(f"Fetching {name}...")

            create_rcv1(data_home, num_targets=40, vocab_size=15000, num_train=10000, num_test=2000)


def build_meta(name: str, kind: str, num_labels: int):
    """Builds dict with meta info to be saved alongside dataset

    Parameters
    ----------
    name : str
        Name of the dataset
    vocab_size : int
        Size of the vocabulary used to encode the dataset
    num_documents : int
        Number of the documents processed
    kind : str
        Value from {unlabelled, single_labeled, multiple_labeled}
    num_labels : int
        Number of labels if applicable

    Returns
    -------
    meta_info : dict
        Dict object with passed meta info, to be dumped as json

    """

    if kind not in {"unlabelled", "single_labeled", "multiple_labeled"}:
        raise ValueError(f"Illegal kind {kind}")

    meta_info = {
        "name": name,
        "kind": kind,
        "num_labels": num_labels
    }

    return meta_info


def fetch_20ng(vocab_size: int, name: str = "20ng"):
    """Fetches and saves 20ng dataset with train/test split and targets to data_home/20ng"""
    try:
        data_home = load_config()["model"]["data_home"]
        dest = f"{data_home}/{name}"

        try:
            os.mkdir(f"{data_home}/{name}")
        except FileExistsError:
            pass

        print("Fetching 20ng...")
        train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

        print("Vectorizing...")
        v = TfidfVectorizer(stop_words="english", max_features=vocab_size)

        # Scipy sparse matrices
        sparse_train_tfidf = v.fit_transform(train.data)
        sparse_test_tfidf = v.transform(test.data)

        print("Saving dataset...")
        with h5py.File(f"{dest}/data.hdf5", "w") as hf:
            hf.create_dataset(name="train", data=sparse_train_tfidf.toarray(), compression="gzip")
            hf.create_dataset(name="train_labels", data=train.target)
            hf.create_dataset(name="test", data=sparse_test_tfidf.toarray(), compression="gzip")
            hf.create_dataset(name="test_labels", data=test.target)

        with open(f"{dest}/meta.json", "w") as f:
            json.dump(build_meta("20ng", kind="single_labeled", num_labels=1), f)

    except (KeyError, IOError):
        print("Couldn't read config.json file")


# TODO
def create_rcv1(data_home: Union[str, os.PathLike], num_targets: int, vocab_size: int,
                num_train: int, num_test: int):
    rcv1 = fetch_rcv1()

    feature_indices = np.argsort(-rcv1.target.sum(axis=0), axis=1)[0, :num_targets]
    feature_indices = np.asarray(feature_indices).squeeze()
    targets = rcv1.target[:, feature_indices]

    word_indices = np.argsort(-rcv1.data.sum(axis=0), axis=1)[0, :vocab_size]
    word_indices = np.asarray(word_indices).squeeze()
    documents = rcv1.data[:, word_indices]

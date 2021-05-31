import os
import shutil
from typing import Optional

import h5py
import numpy as np
import scipy.sparse.csr
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from controllers.usersetup import load_config
from storage.MetaInfo import DatasetMetaInfo

AVAILABLE_DATASETS = ["20ng"]


def extract_train(dataset_name: str) -> Optional[np.ndarray]:
    """Extracts train subset of a given dataset if available

    Parameters
    ----------
    dataset_name : str
        Qualified dataset name

    Returns
    -------
    Optional[np.ndarray]
        The train subset of the dataset, a numpy ndarray with tfidf vectors as rows

    """
    data_home = load_config()["model"]["data_home"]

    try:
        with h5py.File(f"{data_home}/{dataset_name}/data.hdf5", "r") as hf:
            train: np.ndarray = hf["train"][:]
            return train
    except (IOError, OSError):
        print(f"Cannot reach data.hdf5 in {dataset_name}")
        return None


def copy_vectorizer(dataset_name: str, dest_model_name: str) -> bool:
    """Copies the vectorizer from a given dataset to destination model

    Parameters
    ----------
    dataset_name : str
        Qualified dataset name
    dest_model_name : str
        Qualified model name

    Returns
    -------
    bool
        True if the copy succeeded, False if the dataset doesn't have a vectorizer or its vectorizer is unavailable

    """
    config = load_config()
    data_home = config["model"]["data_home"]
    model_home = config["model"]["model_home"]

    src = f"{data_home}/{dataset_name}/vectorizer.pkl"
    dest = f"{model_home}\\{dest_model_name}\\vectorizer.pkl"

    try:
        shutil.copyfile(src, dest)
        return True
    except FileNotFoundError:
        print(f"Dataset {dataset_name} has no defined vectorizer. It will not be usable for searching")
        return False


def create_20ng(vocab_size: int, name: str = "20ng"):
    """Fetches 20ng dataset in plaintext thanks to sklearn, then uses custom Tfidf vectorizer
    to encode the dataset according to specified vocab_size

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size of the dataset
    name : str
        Output name of the dataset, default '20ng'

    Returns
    -------
    None

    """
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
        sparse_train_tfidf: scipy.sparse.csr.csr_matrix = v.fit_transform(train.data)
        sparse_test_tfidf = v.transform(test.data)

        print("Saving dataset...")
        with h5py.File(f"{dest}/data.hdf5", "w") as hf:
            hf.create_dataset(name="train", data=sparse_train_tfidf.toarray(), compression="gzip")
            hf.create_dataset(name="train_labels", data=train.target)
            hf.create_dataset(name="test", data=sparse_test_tfidf.toarray(), compression="gzip")
            hf.create_dataset(name="test_labels", data=test.target)

        mi = DatasetMetaInfo(name,
                             vocab_size,
                             num_train=sparse_train_tfidf.shape[0],
                             num_test=sparse_test_tfidf.shape[0],
                             num_labels=1)

        mi.dump(dest)

    except (KeyError, IOError):
        print("Couldn't read config.json file")

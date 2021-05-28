import datetime
import json
import os
import pickle
import shutil

import pandas as pd

import h5py
import numpy as np
import scipy.sparse.csr
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer

from controllers.usersetup import load_config
from storage.MetaInfo import DatasetMetaInfo

AVAILABLE_DATASETS = ["20ng", "rcv1"]

"""
def create_user_dataset(root_dir: Union[str, os.PathLike], vocab_size: int, name: str, file_ext: str = ""):
    data_home = load_config()["model"]["data_home"]
    dest = f"{data_home}/{name}"

    try:
        os.mkdir(f"{data_home}/{name}")
    except FileExistsError:
        pass

    print("Getting paths...")
    paths = storage.get_paths(root_dir, filter_extension=file_ext)
    print(f"Found {len(paths)} suitable files at {root_dir}")

    print("Vectorizing...")
    v = DocumentVectorizer(vocab_size)
    X, words = v.fit_transform(paths)

    print("Saving dataset...")
    with h5py.File(f"{dest}/data.hdf5", "w") as hf:
        hf.create_dataset(name="train", data=X.toarray(), compression="gzip")

    storage.save_vectorizer(v, dirpath=f"{dest}")

    mi = DatasetMetaInfo(name,
                         vocab_size,
                         num_train=X.shape[0],
                         num_test=0,
                         num_labels=0,
                         user=True,
                         source_dir=root_dir)

    mi.dump(dest)
"""


def extract_train(dataset_name: str):
    data_home = load_config()["model"]["data_home"]

    try:
        with h5py.File(f"{data_home}/{dataset_name}/data.hdf5", "r") as hf:
            train: np.ndarray = hf["train"][:]
            return train
    except IOError:
        print(f"Cannot reach data.hdf5 in {dataset_name}")
        return None


def extract_vectorizer(dataset_name: str):
    data_home = load_config()["model"]["data_home"]

    try:
        with open(f"{data_home}/{dataset_name}/vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)
            return vec
    except IOError:
        print(f"Cannot reach vectorizer in {dataset_name}")
        return None


def copy_vectorizer(dataset_name: str, dest_model_name: str):
    config = load_config()
    data_home = config["model"]["data_home"]
    model_home = config["model"]["model_home"]

    src = f"{data_home}/{dataset_name}/vectorizer.pkl"
    dest = f"{model_home}\\{dest_model_name}\\vectorizer.pkl"

    try:
        shutil.copyfile(src, dest)
    except FileNotFoundError:
        print(f"Dataset {dataset_name} has no defined vectorizer. It will not be usable for searching")


def create_20ng(vocab_size: int, name: str = "20ng"):
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
        sparse_train_tfidf: scipy.sparse.csr.csr_matrix = v.fit_transform(train.data)
        sparse_test_tfidf = v.transform(test.data)

        print("Saving dataset...")
        with h5py.File(f"{dest}/data.hdf5", "w") as hf:
            hf.create_dataset(name="train", data=sparse_train_tfidf.toarray(), compression="gzip")
            hf.create_dataset(name="train_labels", data=train.target)
            hf.create_dataset(name="test", data=sparse_test_tfidf.toarray(), compression="gzip")
            hf.create_dataset(name="test_labels", data=test.target)

        mi = DatasetMetaInfo("20ng",
                             vocab_size,
                             num_train=sparse_train_tfidf.shape[0],
                             num_test=sparse_test_tfidf.shape[0],
                             num_labels=1)

        mi.dump(dest)

    except (KeyError, IOError):
        print("Couldn't read config.json file")


def create_rcv1(vocab_size: int, num_train: int = 100000, num_test: int = 20000, num_labels: int = 40,
                name: str = "rcv1", remove_short: bool = True, remove_long: bool = True):
    """TODO multi label dataset fetching"""
    try:
        data_home = load_config()["model"]["data_home"]
        dest = f"{data_home}/{name}"

        try:
            os.mkdir(f"{data_home}/{name}")
        except FileExistsError:
            pass

        print("Fetching rcv1...")
        rcv1 = fetch_rcv1()

        feature_indices = np.argsort(-rcv1.target.sum(axis=0), axis=1)[0, :num_labels]
        feature_indices = np.asarray(feature_indices).squeeze()
        targets = rcv1.target[:, feature_indices]

        word_indices = np.argsort(-rcv1.data.sum(axis=0), axis=1)[0, :vocab_size]
        word_indices = np.asarray(word_indices).squeeze()
        docs = rcv1.data[:, word_indices]

        targets = [t for t in targets]
        documents = [d for d in docs]

        df = pd.DataFrame({'doc_id': rcv1.sample_id.tolist(), 'bow': documents, 'label': targets})
        df.set_index('doc_id', inplace=True)
        print('total docs: {}'.format(len(df)))

        # remove any empty labels
        def count_num_tags(target):
            return target.sum()

        def get_num_word(bow):
            return bow.count_nonzero()

        df = df[df.label.apply(count_num_tags) > 0]
        print('after filter: total docs: {}'.format(len(df)))

        df = df[df.bow.apply(get_num_word) > 0]
        print('after filter: total docs: {}'.format(len(df)))

        # remove any empty documents
        if remove_short:
            print('remove any short document that has less than 5 words.')
            df = df[df.bow.apply(get_num_word) > 5]
            print('num docs: {}'.format(len(df)))

        if remove_long:
            print('remove any long document that has more than 500 words.')
            df = df[df.bow.apply(get_num_word) <= 500]
            print('num docs: {}'.format(len(df)))

        df = df.reindex(np.random.permutation(df.index))

        sampled_df = df.sample(num_train + num_test)
        train_df = sampled_df.iloc[:num_train]

        test_df = sampled_df.iloc[num_train:]
        cv_df = test_df[:num_test // 2]
        test_df = test_df[num_test // 2:]

        train = train_df.to_numpy()
        test = test_df.to_numpy()

        # What is this returning

        mi = DatasetMetaInfo(name, vocab_size, num_train, num_test, num_labels)
        mi.dump(dest)

    except (KeyError, IOError):
        print("Couldn't read config.json file")

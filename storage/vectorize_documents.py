import os
import pickle
from typing import Union

import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class DocumentVectorizer:
    """Wrapper for scipy TfidfVectorizer with additional save/load functionalities"""

    def __init__(self, vocab_size: int, name: str = "vectorizer", stop_words="english", decode_errors="ignore"):
        """
        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary to use when encoding
        stop_words : list[str] or str
            List of stopwords, or {english} for english stopwords
        decode_errors : str
            {strict, ignore, replace}, strict raises when encountered non unicode file

        """
        self.model = TfidfVectorizer(input="filename",
                                     decode_error=decode_errors,
                                     stop_words=stop_words,
                                     max_features=vocab_size)
        self.vocab_size = vocab_size
        self.stop_words = stop_words
        self.decode_errors = decode_errors
        self.name = name

    def fit_transform(self, paths: list[str]):
        """Fits the model and transforms paths into scipy matrix and list of features names (words)"""
        X = self.model.fit_transform(paths)
        return X, self.model.get_feature_names()

    def transform(self, paths: list[str]):
        """Uses the fitted model to transform paths into scipy matrix"""
        print(f"Running transform on {len(paths)} paths")
        X = self.model.transform(paths)
        return X

    @staticmethod
    def save_transform(X, words: list[str], path: Union[str, os.PathLike], dataset_name: str,
                       x_name="X", words_name="vocab"):
        """TODO check if used
        Saves wcv matrix as .h5 ndarray and words as .pkl file

        Parameters
        ----------
        X : scipy sparse matrix
            Matrix of encoded word count vectors, assumed to be TF-IDF
        words : list[str]
            List of feature names used for the encoding
        path : str
            PathLike string, directory in which the data will be saved
        dataset_name : str
            Name of the output dir containing data
        x_name
            Name of the output X file name
        words_name
            Name of the output words file name

        Raises
        -------
        OSError
            When the os module cannot create dir at 'path/dataset_name'
        FileExistsError
            When dir already exists at 'path/dataset_name'

        """
        dirpath = f"{path}/{dataset_name}"
        x_path = f"{dirpath}/{x_name}.h5"
        words_path = f"{dirpath}/{words_name}.pkl"

        os.mkdir(dirpath)

        with h5py.File(x_path, "w") as hf:
            hf.create_dataset(name=dataset_name, data=X.toarray())

        with open(words_path, "wb") as f:
            pickle.dump(words, f)


def save_vectorizer(vectorizer: DocumentVectorizer, dirpath: Union[str, os.PathLike]):
    """Saves a DocumentVectorizer as .pkl object at specified path, overwrites any underlying model"""
    with open(dirpath + "/" + vectorizer.name + ".pkl", "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(dirpath: Union[str, os.PathLike], name: str = "vectorizer.pkl"):
    """Returns a DocumentVectorizer loaded from file at dirpath/name"""
    with open(dirpath + "/" + name, "rb") as f:
        print(dirpath)
        data: DocumentVectorizer = pickle.load(f)
        return data


def get_paths(root_dir: Union[str, os.PathLike], filter_extension: str = None):
    """Returns absolute paths at root_dir and all subdirs with a given file format, None includes all file extensions

    Parameters
    ----------
    root_dir : str or os.PathLike
        either relative or absolute path to a root_dir containing text files
    filter_extension : str
        consider only files with specified extension, None removes any filtering

    Returns
    -------
    list[str]
        list of absolute filepaths of files with specified extension at root_dir and all subdirs

    """
    result = []
    for dirpath, _, files in os.walk(os.path.abspath(root_dir)):
        for f in tqdm(files):
            abspath = os.path.join(dirpath, f)
            if filter_extension:
                ext = os.path.splitext(abspath)[-1].lower()
                if ext == filter_extension:
                    result.append(abspath)
            else:
                result.append(abspath)
    return result

import os

from sklearn.feature_extraction.text import TfidfVectorizer


def get_paths(root_dir: str, filter_extension: str = None):
    """Returns absolute paths at root_dir and all subdirs with a given file format, None includes all file extensions

    Parameters
    ----------
    root_dir : str
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
        for f in files:
            abspath = os.path.join(dirpath, f)
            if filter_extension:
                ext = os.path.splitext(abspath)[-1].lower()
                if ext == filter_extension:
                    result.append(abspath)
            else:
                result.append(abspath)
    return result


def vectorize_documents(paths: list[str], vocab_size: int, stop_words="english", decode_errors="ignore"):
    """Vectorizes documents at given paths to .txt files into sparse TF-IDF matrix

    Parameters
    ----------
    paths : list[str]
        List of filepaths to fit
    vocab_size : int
        Size of the vocabulary to use when encoding
    stop_words : list[str] or str
        List of stopwords, or {english} for english stopwords
    decode_errors : str
        {strict, ignore, replace}, strict raises when encountered non unicode file

    Returns
    -------
    X, words : tuple[sparse matrix, list[str]]
        scipy sparse matrix size (len(paths), vocab_size); list of feature names

    """
    vectorizer = TfidfVectorizer(input="filename",
                                 decode_error=decode_errors,
                                 stop_words=stop_words,
                                 max_features=vocab_size)
    X = vectorizer.fit_transform(paths)
    return X, vectorizer.get_feature_names()

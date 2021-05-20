import os

from sklearn.feature_extraction.text import TfidfVectorizer


def get_paths(root_dir: str, filter_extension: str = None):
    """Returns absolute paths at root_dir and all subdirs with a given file format, None includes all file extensions

    Args:
        root_dir: either relative or absolute path to a root_dir containing text files
        filter_extension: consider only files with specified extension, None removes any filtering

    Returns:
        list[str]: a list of absolute filepaths found at root_dir and all subdirs
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


def vectorize_documents(paths: list[str], vocab_size: int, stop_words):
    """Vectorizes documents at given paths to .txt files into sparse TF-IDF matrix

    Args:
        paths: a list of absolute paths to .txt files
        vocab_size: size of the vocab to vectorize by
        stop_words: string 'english' for english stopwords or custom list[str] of stopwords

    Returns:
        tuple[sparse matrix, list[str]]: encoded documents and the list of the feature names (i.e. words)
    """
    vectorizer = TfidfVectorizer(decode_error="ignore", stop_words=stop_words, max_features=vocab_size)
    X = vectorizer.fit_transform(paths)
    return X, vectorizer.get_feature_names()

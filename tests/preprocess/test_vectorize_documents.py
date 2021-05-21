import os

import pytest

from preprocess.vectorize_documents import get_paths, vectorize_documents, save_vectorized


@pytest.fixture
def no_such_dir():
    return "no_such_dir"


@pytest.fixture
def valid_dir():
    return "dataset_mock"


@pytest.fixture
def cwd():
    return os.getcwd()


@pytest.fixture
def pangram_words():
    return ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog"]


@pytest.fixture
def pangram_no_stopwords():
    return ["quick", "brown", "fox", "jumps", "lazy", "dog"]


def test_get_paths_no_root_dir(no_such_dir):
    actual = get_paths(f"resources/{no_such_dir}")
    assert actual == []


def test_get_paths_valid_root(cwd, valid_dir):
    path = f"resources/{valid_dir}"
    actual = get_paths(path)
    expected = [f'{cwd}\\resources\\dataset_mock\\file01.txt',
                f'{cwd}\\resources\\dataset_mock\\nested\\file02.txt',
                f'{cwd}\\resources\\dataset_mock\\file03',
                f'{cwd}\\resources\\dataset_mock\\file04.pkl',
                f'{cwd}\\resources\\dataset_mock\\nested\\file05.npy',
                f'{cwd}\\resources\\dataset_mock\\file06.mat']
    assert sorted(actual) == sorted(expected)


def test_get_paths_valid_root_filter_txt(cwd, valid_dir):
    path = f"resources/{valid_dir}"
    actual = get_paths(path, filter_extension=".txt")
    expected = [f'{cwd}\\resources\\dataset_mock\\file01.txt',
                f'{cwd}\\resources\\dataset_mock\\nested\\file02.txt']
    assert sorted(actual) == sorted(expected)


def test_get_paths_valid_root_filter_no_ext(cwd, valid_dir):
    path = f"resources/{valid_dir}"
    actual = get_paths(path, filter_extension=".mat")
    expected = [f'{cwd}\\resources\\dataset_mock\\file06.mat']
    assert sorted(actual) == sorted(expected)


def test_vectorize_documents_empty_paths():
    """No file -> empty vocab"""
    with pytest.raises(ValueError):
        X, words = vectorize_documents([], 100)


def test_vectorize_documents_invalid_path():
    """Invalid filename among paths"""
    with pytest.raises(FileNotFoundError):
        X, words = vectorize_documents(["resources/dataset_mock/no_such_file.txt"], 100)


def test_vectorize_documents_valid_path_vocab_size_negative():
    with pytest.raises(ValueError):
        X, words = vectorize_documents(["resources/dataset_mock/file01.txt"], -45)


def test_vectorize_documents_valid_pangram(pangram_words):
    """Tokens are alphanumeric words length 2 or more -> 'a' is not a token"""
    X, words = vectorize_documents(["resources/dataset_mock/file01.txt"], 100, stop_words=[])
    pangram_words.remove("a")
    assert sorted(words) == sorted(pangram_words)


def test_vectorize_documents_valid_pangram_stopwords_removed(pangram_no_stopwords):
    X, words = vectorize_documents(["resources/dataset_mock/file01.txt"], 100, stop_words="english")
    assert sorted(words) == sorted(pangram_no_stopwords)


def test_vectorize_documents_binary_file(pangram_no_stopwords):
    """Strict decoding raises error when file is not UTF-8, ignore reads the file contents anyway"""
    with pytest.raises(UnicodeDecodeError):
        X, words = vectorize_documents(["resources/dataset_mock/file04.pkl"], 100, decode_errors="strict")

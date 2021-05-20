import os

import pytest

from preprocess.vectorize_documents import get_paths


@pytest.fixture
def no_such_dir():
    return "no_such_dir"


@pytest.fixture
def valid_dir():
    return "dataset_mock"


@pytest.fixture
def cwd():
    return os.getcwd()


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

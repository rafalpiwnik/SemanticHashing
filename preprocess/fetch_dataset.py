import os
from typing import Union

import sklearn.utils
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1

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
            print(f"Dataset root for {name} already exists")

    dest = f"{data_home}/{name}"
    if name == "20ng":
        # TODO progress bar as number of files ~18k here
        return fetch_20newsgroups(data_home=dest)
    elif name == "rcv1":
        return fetch_rcv1(data_home=dest)

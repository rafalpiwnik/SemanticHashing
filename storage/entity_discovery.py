import os
from glob import glob

import numpy as np

from controllers import usersetup
from storage.MetaInfo import DatasetMetaInfo, ModelMetaInfo

META_FILE_NAME = "meta.json"
DATA_FILE_NAME = "data.hdf5"


def scan_datasets() -> list[DatasetMetaInfo]:
    """Returns the list of DatasetMetaInfo for dirs in data_home,
     if there is no meta.json provided name and path is inferred"""
    data_home = usersetup.load_config()["model"]["data_home"]

    result: list[DatasetMetaInfo] = list()

    # Get the list of paths and dataset names, skip the first dir (the parent dir)
    datasets = os.walk(data_home)
    next(datasets)

    for d in datasets:
        path, _, files = d

        name = path.split("\\")[-1]

        if META_FILE_NAME in files and DATA_FILE_NAME in files:
            mi = DatasetMetaInfo.from_file(path)
        elif DATA_FILE_NAME in files:
            mi = DatasetMetaInfo(name, np.NAN, np.NAN, np.NAN)
        else:
            mi = DatasetMetaInfo.undefined_preset(name)

        result.append(mi)

    return result


def scan_models() -> list[ModelMetaInfo]:
    """Returns the list of ModelMetaInfo for dirs in model_home,
    if there is no meta.json provided name and path is inferred"""
    model_home = usersetup.load_config()["model"]["model_home"]

    result: list[ModelMetaInfo] = list()

    # Get the list of paths and dataset names, skip the first dir (the parent dir)
    model_paths = glob(f"{model_home}/*/")

    for path in model_paths:
        name = path.split("\\")[-2]

        try:
            mi = ModelMetaInfo.from_file(path)
            result.append(mi)
        except FileNotFoundError:
            mi = ModelMetaInfo(name)
            result.append(mi)

    return result

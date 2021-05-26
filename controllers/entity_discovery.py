import os

import numpy as np

from controllers import usersetup
from preprocess.MetaInfo import DatasetMetaInfo

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

        print(path)
        print(files)

        name = path.split("\\")[-1]

        if META_FILE_NAME in files and DATA_FILE_NAME in files:
            mi = DatasetMetaInfo.from_file(path)
        elif DATA_FILE_NAME in files:
            mi = DatasetMetaInfo(name, np.NAN, np.NAN, np.NAN)
        else:
            mi = DatasetMetaInfo.undefined_preset(name)

        result.append(mi)

    return result

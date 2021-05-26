import os

import numpy as np

from controllers import usersetup
from preprocess.MetaInfo import DatasetMetaInfo

META_FILE_NAME = "meta.json"


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

        if META_FILE_NAME in files:
            result.append(DatasetMetaInfo.from_file(path))
        else:
            mi = DatasetMetaInfo(name, np.NAN, np.NAN, np.NAN)
            mi.set_source_dir(path)
            result.append(mi)

    return result

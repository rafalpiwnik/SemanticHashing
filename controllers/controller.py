import os
import shutil
from typing import Union

import h5py
from PyQt5 import QtWidgets

import storage
from controllers.usersetup import load_config
from gui.DatasetWidget import DatasetWidget
from gui.ModelWidget import ModelWidget
from storage import DocumentVectorizer, datasets
from storage.MetaInfo import DatasetMetaInfo
from storage.entity_discovery import scan_models, scan_datasets


def fetch_datasets_to_widgets() -> list[DatasetWidget]:
    result = []
    mi_datasets = scan_datasets()

    for mi in mi_datasets:
        d = DatasetWidget()
        d.set_fields(mi)
        result.append(d)

    return result


def fetch_models_to_widgets() -> list[ModelWidget]:
    result = []
    mi_datasets = scan_models()

    for mi in mi_datasets:
        d = ModelWidget()
        d.set_fields(mi)
        result.append(d)

    return result


def check_dataset_available(name: str):
    """Checks if dataset exits"""
    data_home = load_config()["model"]["data_home"]
    dest = f"{data_home}/{name}"

    try:
        return DatasetMetaInfo.from_file(dest)
    except OSError:
        return None



def check_model_exists(name: str):
    """Checks if dataset exits"""
    model_home = load_config()["model"]["model_home"]
    dest = f"{model_home}/{name}"

    return os.path.exists(dest)


def rename_dataset(name_old: str, name_new: str):
    data_home = load_config()["model"]["data_home"]
    src_dirpath = f"{data_home}/{name_old}"
    dest_dirpath = f"{data_home}/{name_new}"
    # TODO
    # os.rename(src_dirpath, dest_dirpath)


def remove_dataset(name: str):
    data_home = load_config()["model"]["data_home"]
    dirpath = f"{data_home}/{name}"

    try:
        shutil.rmtree(dirpath)
        return True
    except OSError:
        print("Could not delete")
        return False


def create_user_dataset(root_dir: Union[str, os.PathLike], vocab_size: int, name: str, file_ext: str = "",
                        progbar: QtWidgets.QProgressBar = None):
    """Creates a user dataset, fits a vectorizer and saves it

    Parameters
    ----------
    root_dir : Union[str, os.PathLike]
        Root dir of the text files to vectorize
    vocab_size : int
        Size of the vocabulary to consider
    name :
        Name of the dataset
    progbar
    file_ext :
        Consider only files with a given extension


    Returns
    -------
    None
        Saves the dataset and the vectorizer to data_home

    """
    if progbar:
        progbar.setValue(7)

    data_home = load_config()["model"]["data_home"]
    dest = f"{data_home}/{name}"

    try:
        os.mkdir(f"{data_home}/{name}")
    except FileExistsError:
        pass

    if progbar:
        progbar.setValue(15)

    print("Getting paths...")
    paths = storage.get_paths(root_dir, filter_extension=file_ext)
    print(f"Found {len(paths)} suitable files at {root_dir}")

    if progbar:
        progbar.setValue(35)

    print("Vectorizing...")
    v = DocumentVectorizer(vocab_size)
    X, words = v.fit_transform(paths)

    if progbar:
        progbar.setValue(95)

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

    if progbar:
        progbar.setValue(100)

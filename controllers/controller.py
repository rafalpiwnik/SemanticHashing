import os
import shutil
from typing import Union, Optional

import vdsh.utility
from controllers.usersetup import load_config
from gui.DatasetWidget import DatasetWidget
from gui.ModelWidget import ModelWidget
from storage.MetaInfo import DatasetMetaInfo, ModelMetaInfo
from storage.entity_discovery import scan_models, scan_datasets


def fetch_datasets_to_widgets() -> list[DatasetWidget]:
    """Fetches datasets from data_home and returns them as DatasetWidgets"""
    result = []
    mi_datasets = scan_datasets()

    for mi in mi_datasets:
        d = DatasetWidget()
        d.set_fields(mi)
        result.append(d)

    return result


def fetch_models_to_widgets() -> list[ModelWidget]:
    """Fetches models from model_home and returns them as ModelWidgets"""
    result = []
    mi_datasets = scan_models()

    for mi in mi_datasets:
        d = ModelWidget()
        d.set_fields(mi)
        result.append(d)

    return result


def check_dataset_available(name: str) -> Optional[DatasetMetaInfo]:
    """Returns dataset meta info if it exists, else None"""
    data_home = load_config()["model"]["data_home"]
    dest = f"{data_home}/{name}"

    try:
        return DatasetMetaInfo.from_file(dest)
    except OSError:
        return None


def check_model_available(name: str) -> Optional[ModelMetaInfo]:
    """Returns model meta info if it exists, else None"""
    model_home = load_config()["model"]["model_home"]
    dest = f"{model_home}/{name}"

    try:
        return ModelMetaInfo.from_file(dest)
    except OSError:
        return None


def check_model_has_vectorizer(model_name: str):
    """Returns True if specified model has a vectorizer assigned, False otherwise"""
    data_home = load_config()["model"]["model_home"]

    return os.path.isfile(f"{data_home}/{model_name}/vectorizer.pkl")


def remove_dataset(name: str) -> bool:
    """Permanently removes the dataset from files at data_home

    Parameters
    ----------
    name : str
        Qualified dataset name

    Returns
    -------
    bool
        True if deletion succeeded, False otherwise

    """
    data_home = load_config()["model"]["data_home"]
    dirpath = f"{data_home}/{name}"
    return _remove_entity(dirpath)


def remove_model(name: str) -> bool:
    """Permanently removes the model from files at model_home

        Parameters
        ----------
        name : str
            Qualified model name

        Returns
        -------
        bool
            True if deletion succeeded, False otherwise

        """
    model_home = load_config()["model"]["model_home"]
    dirpath = f"{model_home}/{name}"
    return _remove_entity(dirpath)


def _remove_entity(dirpath: Union[str, os.PathLike]) -> bool:
    """Permanently removes dir and all its contents at dirpath, returns True if action succeeded, False otherwise"""
    try:
        shutil.rmtree(dirpath)
        return True
    except OSError:
        print("Could not delete")
        return False


def create_model(vocab_dim: int, hidden_dim: int, latent_dim: int, kl_step: float, dropout_prob: float,
                 name: str = "default"):
    """Creates a VDSH model and saves it to model_home"""
    model = vdsh.utility.create_vdsh(vocab_dim, hidden_dim, latent_dim, kl_step, dropout_prob, name)
    vdsh.utility.dump_model(model)

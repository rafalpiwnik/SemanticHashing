import os
import shutil
from typing import Union

import h5py
from PyQt5 import QtWidgets

import storage
import vdsh.utility
from controllers.usersetup import load_config
from gui.DatasetWidget import DatasetWidget
from gui.ModelWidget import ModelWidget
from storage import DocumentVectorizer
from storage.MetaInfo import DatasetMetaInfo, ModelMetaInfo
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
    """Returns dataset meta info if it exists, else None"""
    data_home = load_config()["model"]["data_home"]
    dest = f"{data_home}/{name}"

    try:
        return DatasetMetaInfo.from_file(dest)
    except OSError:
        return None


def check_model_available(name: str):
    """Returns model meta info if it exists, else None"""
    model_home = load_config()["model"]["model_home"]
    dest = f"{model_home}/{name}"

    try:
        return ModelMetaInfo.from_file(dest)
    except OSError:
        return None


def remove_dataset(name: str):
    data_home = load_config()["model"]["data_home"]
    dirpath = f"{data_home}/{name}"

    try:
        shutil.rmtree(dirpath)
        return True
    except OSError:
        print("Could not delete")
        return False


def remove_model(name: str):
    model_home = load_config()["model"]["model_home"]
    dirpath = f"{model_home}/{name}"

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

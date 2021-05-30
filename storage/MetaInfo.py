import datetime
import json
import os
from typing import Union

import numpy as np


class DatasetMetaInfo:
    """Meta info dict to be saved with the dataset to provide additional info"""

    def __init__(self, name: str, vocab_size: int = np.NAN, num_train: int = np.NAN, num_test: int = 0,
                 num_labels: int = 0,
                 user: bool = False,
                 source_dir: Union[str, os.PathLike] = None):
        super().__init__()

        self.info = {
            "name": name,
            "user_author": user,
            "source_dir": source_dir,
            "vocab_size": vocab_size,
            "num_train": num_train,
            "num_test": num_test,
            "num_labels": num_labels,
            "time_saved": ""
        }

    @property
    def name(self):
        return self.info["name"]

    @property
    def vocab_size(self):
        return self.info["vocab_size"]

    def set_source_dir(self, dirpath: Union[str, os.PathLike]):
        self.info["source_dir"] = str(dirpath)

    @classmethod
    def from_file(cls, dirpath: Union[str, os.PathLike]):
        with open(dirpath + "/meta.json", "r") as f:
            result = cls("loaded")
            result.info = json.load(f)
            return result

    @classmethod
    def undefined_preset(cls, name: str):
        result = cls(name)
        result.info = {"name": name}
        return result

    def dump(self, dirpath: Union[str, os.PathLike]):
        date = datetime.datetime.now()
        date_iso = date.isoformat()
        self.info["time_saved"] = date_iso
        with open(dirpath + "/meta.json", "w") as f:
            json.dump(self.info, f)


class ModelMetaInfo:
    """Meta info dict to be saved with the model to provide additional info"""

    def __init__(self, name: str, vocab_size: int = np.NAN, hidden_dim: int = np.NAN, latent_dim: int = np.NAN,
                 kl_step: float = np.NAN, dropout_prob: float = np.NAN, fit_dataset: str = "", fit: bool = False):
        super().__init__()

        self.info = {
            "name": name,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "kl_step": kl_step,
            "dropout_prob": dropout_prob,
            "fit": fit,
            "fit_dataset": fit_dataset,
            "fit_time": ""
        }

    def flag_fit(self, dataset_name: str):
        self.info["fit"] = True
        self.info["fit_time"] = datetime.datetime.now().isoformat()
        self.info["fit_dataset"] = dataset_name

    @property
    def vocab_size(self):
        return self.info["vocab_size"]

    @property
    def hidden_dim(self):
        return self.info["hidden_dim"]

    @property
    def latent_dim(self):
        return self.info["latent_dim"]

    @property
    def kl_step(self):
        return self.info["kl_step"]

    @property
    def dropout_prob(self):
        return self.info["dropout_prob"]

    @property
    def name(self):
        return self.info["name"]

    @property
    def dataset_name(self):
        return self.info["fit_dataset"]

    @property
    def is_fit(self):
        return self.info["fit_dataset"] != ""

    def dump(self, dirpath: Union[str, os.PathLike]):
        date = datetime.datetime.now()
        date_iso = date.isoformat()
        self.info["time_saved"] = date_iso
        with open(dirpath + "/meta.json", "w") as f:
            json.dump(self.info, f)

    @classmethod
    def from_file(cls, dirpath: Union[str, os.PathLike]):
        with open(dirpath + "/meta.json", "r") as f:
            result = ModelMetaInfo("loaded")
            result.info = json.load(f)
            return result


def are_compatible(data: DatasetMetaInfo, model: ModelMetaInfo):
    """Returns true iff model vocab_dim and data vocab_dim are a match"""
    return data.vocab_size == model.vocab_size


def are_native(data: DatasetMetaInfo, model: ModelMetaInfo):
    """Returns true if model has been trained on the specified data"""
    return model.dataset_name == data.name

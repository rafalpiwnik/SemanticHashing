import datetime
import json
import os
from typing import Union


class DatasetMetaInfo:
    """Meta info dict to be saved with the dataset to provide additional info"""

    def __init__(self, name: str, num_train: int, num_test: int, num_labels: int = 0, user: bool = False,
                 source_dir: Union[str, os.PathLike] = None):
        super().__init__()
        assert num_train >= 0
        assert num_test >= 0
        assert num_labels >= 0

        self.info = {
            "name": name,
            "user_author": user,
            "source_dir": source_dir,
            "num_train": num_train,
            "num_test": num_test,
            "num_labels": num_labels,
            "time_saved": ""
        }

    @classmethod
    def from_file(cls, dirpath: Union[str, os.PathLike]):
        with open(dirpath + "/meta.json", "r") as f:
            result = cls
            result.info = json.load(f)
            return result

    def dump(self, dirpath: Union[str, os.PathLike]):
        date = datetime.datetime.now()
        date_iso = date.isoformat()
        self.info["time_saved"] = date_iso
        with open(dirpath + "/meta.json", "w") as f:
            json.dump(self.info, f)


class ModelMetaInfo():
    """Meta info dict to be saved with the model to provide additional info"""

    def __init__(self, name: str, vocab_size: int, hidden_dim: int, latent_dim: int, kl_step: float,
                 dropout_prob: float, fit_dataset: str = "", fit: bool = False):
        super().__init__()
        assert vocab_size >= 0
        assert hidden_dim >= 0
        assert latent_dim >= 0

        self.info = {
            "name": name,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "kl_step": kl_step,
            "dropout_prob": dropout_prob,
            "fit": fit,
            "epochs": 0,
            "fit_dataset": fit_dataset,
            "fit_time": ""
        }

    def flag_fit(self, dataset_name: str):
        self.info["fit"] = True
        self.info["fit_time"] = datetime.datetime.now().isoformat()
        self.info["fit_dataset"] = dataset_name

    def set_epochs(self, num_epochs: int):
        self.info["epochs"] = num_epochs

    @property
    def name(self):
        return self.info["name"]

    @property
    def dataset_name(self):
        return self.info["fit_dataset"]

    def dump(self, dirpath: Union[str, os.PathLike]):
        date = datetime.datetime.now()
        date_iso = date.isoformat()
        self.info["time_saved"] = date_iso
        with open(dirpath + "/meta.json", "w") as f:
            json.dump(self.info, f)

    @classmethod
    def from_file(cls, dirpath: Union[str, os.PathLike]):
        with open(dirpath + "/meta.json", "r") as f:
            result = cls
            result.info = json.load(f)
            return result

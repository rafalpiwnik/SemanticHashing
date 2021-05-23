import os
import shutil
from typing import Union

CONFIG_ROOT = ".semhash"
DIRS = ["model_home", "data_home", "logs"]
CONFIG_NAME = "config.json"


def create_homedir(home_path: Union[str, os.PathLike] = os.path.expanduser("~"), overwrite: bool = False):
    try:
        os.mkdir(f"{home_path}/{CONFIG_ROOT}")
    except FileExistsError:
        pass

    if overwrite:
        with open(f"{home_path}/{CONFIG_NAME}", "w") as f:
            f.write("KAPPA")
    else:
        pass


def remove_homedir(home_path: Union[str, os.PathLike] = os.path.expanduser("~")):
    shutil.rmtree(f"{home_path}/{CONFIG_ROOT}")

import json
import os
import shutil
from typing import Union

CONFIG_ROOT = ".semhash"
DIRS = ["model_home", "data_home", "logs"]
CONFIG_NAME = "config.json"

LOCAL_CONFIG_PATH = "../resources/config.json"

DEFAULT_SETTINGS = {
    "model": {
        "model_home": "",
        "data_home": "",
        "logs_home": ""
    },
    "gui": {
        "geometry": "100,200,350,350"
    }
}


def load_config(home_path: Union[str, os.PathLike] = os.path.expanduser("~")):
    """Loads config.json at ~/.semhash, returns json read as dict"""
    with open(f"{home_path}/{CONFIG_ROOT}/{CONFIG_NAME}", "r") as f:
        settings = json.load(f)
        return settings


def config_exists(home_path: Union[str, os.PathLike] = os.path.expanduser("~")):
    """Checks if config.json and underlying dir exist in ~"""
    ok = os.path.isdir(f"{home_path}/{CONFIG_ROOT}") and os.path.isfile(f"{home_path}/{CONFIG_ROOT}/{CONFIG_NAME}")
    return ok


def setup_homedir(home_path: Union[str, os.PathLike] = os.path.expanduser("~"), overwrite: bool = False):
    """Setups homedir, overwrite flag as safeguard"""
    has_config = config_exists(home_path)
    if not has_config or (has_config and overwrite):
        try:
            os.mkdir(f"{home_path}/{CONFIG_ROOT}")
        except FileExistsError:
            print("\tAlready exists: config dir...")

        with open(f"{home_path}/{CONFIG_ROOT}/{CONFIG_NAME}", "w") as f:
            # manual
            DEFAULT_SETTINGS["model"]["data_home"] = f"{home_path}/{CONFIG_ROOT}/data_home"
            DEFAULT_SETTINGS["model"]["model_home"] = f"{home_path}/{CONFIG_ROOT}/model_home"
            DEFAULT_SETTINGS["model"]["logs_home"] = f"{home_path}/{CONFIG_ROOT}/logs"

            json.dump(DEFAULT_SETTINGS, f)
            print("\tOverwriting config.json...")

        for d in DIRS:
            try:
                os.mkdir(f"{home_path}/{CONFIG_ROOT}/{d}")
            except FileExistsError:
                print(f"\tAlready exists: {d}")
    else:
        print("Config exists, overwrite not permitted")


def remove_homedir(home_path: Union[str, os.PathLike] = os.path.expanduser("~")):
    """Removed entire ~/.semhash dir with all underlying files"""
    shutil.rmtree(f"{home_path}/{CONFIG_ROOT}")

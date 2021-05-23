import json
import os
import shutil
from typing import Union

HOME_PATH: Union[str, os.PathLike] = os.path.expanduser("~")

CONFIG_ROOT = "semhash"
DIRS = ["model_home", "data_home", "logs"]
CONFIG_NAME = "config.json"

LOCAL_CONFIG_PATH = "../resources/config.json"

DEFAULT_SETTINGS = {
    "model": {
        "model_home": f"{HOME_PATH}/{CONFIG_ROOT}/model_home",
        "data_home": f"{HOME_PATH}/{CONFIG_ROOT}/data_home",
        "logs_home": f"{HOME_PATH}/{CONFIG_ROOT}/logs"
    },
    "gui": {
        "geometry": ""
    }
}


def load_config(home_path: Union[str, os.PathLike] = os.path.expanduser("~")):
    """Loads config.json at ~/.semhash, returns json read as dict

    Returns
    -------
    settings : dict
        Dictionary with structure as defined in config.json

    Raises
    -------
    IOError
        When config file in unreachable or decoding cannot be completed
    """
    try:
        with open(f"{home_path}/{CONFIG_ROOT}/{CONFIG_NAME}", "r") as f:
            settings = json.load(f)
            return settings
    except json.JSONDecodeError as e:
        raise IOError(f"Cannot read {home_path}/{CONFIG_ROOT}/{CONFIG_NAME} because of JsonDecodeError") from e


def config_exists(home_path: Union[str, os.PathLike] = os.path.expanduser("~")):
    """Checks if config.json and underlying dir exist in ~"""
    ok = os.path.isdir(f"{home_path}/{CONFIG_ROOT}") and os.path.isfile(f"{home_path}/{CONFIG_ROOT}/{CONFIG_NAME}")
    return ok


def setup_homedir(home_path: Union[str, os.PathLike] = HOME_PATH, overwrite: bool = False):
    """Setups homedir, overwrite flag as safeguard"""
    has_config = config_exists(HOME_PATH)
    if not has_config or (has_config and overwrite):
        try:
            os.mkdir(f"{home_path}/{CONFIG_ROOT}")
        except FileExistsError:
            print("\tAlready exists: config dir...")

        with open(f"{home_path}/{CONFIG_ROOT}/{CONFIG_NAME}", "w") as f:
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

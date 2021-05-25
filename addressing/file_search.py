import os
from typing import Union

import preprocess
import vdsh.utility


def search(model_name: str, search_root: Union[str, os.PathLike], example_file_path: Union[str, os.PathLike],
           file_extension: str = ""):
    """Conducts search with a given model on a specified search space with a file given as an example"""
    model, vectorizer = vdsh.utility.load_model(model_name)

    search_paths = preprocess.get_paths(search_root)

    search_target = vectorizer.transform(search_paths)
    example = vectorizer.transform([example_file_path])

    print("SS")

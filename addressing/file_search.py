import os
from typing import Union

import addressing
import storage
import vdsh.utility


# Search schema for presentation
def search(model_name: str, search_root: Union[str, os.PathLike], example_file_path: Union[str, os.PathLike]):
    """Conducts search with a given model on a specified search space with a file given as an example"""
    model, vectorizer = vdsh.utility.load_model(model_name)

    print(model.meta.name)

    search_paths = storage.get_paths(search_root)

    search_target = vectorizer.transform(search_paths).toarray()
    example = vectorizer.transform([example_file_path]).toarray()

    pred_search_target = model.predict(search_target)
    pred_example = model.predict(example)

    codes_search_target = addressing.medhash_transform(pred_search_target)
    code_example = addressing.MedianHash(pred_example)

    retrieved_indices = addressing.top_k_indices(code_example, pool=codes_search_target, k=25)

    retrieved = [search_paths[i] for i in retrieved_indices]

    print(retrieved)

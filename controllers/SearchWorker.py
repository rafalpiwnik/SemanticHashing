import os
from typing import Union

from PyQt5.QtCore import QObject, pyqtSignal

import addressing
import storage
import vdsh

PROGRESS_PATHS_DISCOVERED = 10
PROGRESS_MODEL_LOADED = 17
PROGRESS_VECTORIZER_TRANSFORMED = 53
PROGRESS_PREDICTED = 64
PROGRESS_TRANSFORMED = 79
PROGRESS_FINISHED = 100


class SearchWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    files_retrieved = pyqtSignal(list, list, list)
    status = pyqtSignal(str)

    def __init__(self, model_name: str, search_root: Union[str, os.PathLike], num_results: int,
                 target_path: Union[str, os.PathLike]):
        super().__init__()

        self.model_name = model_name
        self.search_root = search_root
        self.num_results = num_results
        self.target_path = target_path

    def run(self):
        self.status.emit(f"Getting paths in in {self.search_root}...")

        search_paths = storage.get_paths(self.search_root)

        self.status.emit(f"Discovered {len(search_paths)} paths. Loading model and vectorizer...")
        self.progress.emit(PROGRESS_PATHS_DISCOVERED)

        model, vectorizer = vdsh.utility.load_model(self.model_name)

        self.status.emit(f"Loaded model '{model.meta.name}'. Running vectorizer on search_dir...")
        self.progress.emit(PROGRESS_MODEL_LOADED)

        search_target = vectorizer.transform(search_paths).toarray()
        example = vectorizer.transform([self.target_path]).toarray()

        self.status.emit("Files vectorized. Running predict...")
        self.progress.emit(PROGRESS_VECTORIZER_TRANSFORMED)

        pred_search_target = model.predict(search_target)
        pred_example = model.predict(example)

        self.status.emit(f"Creating binary codes...")
        self.progress.emit(PROGRESS_PREDICTED)

        codes_search_target = addressing.medhash_transform(pred_search_target)
        code_example = addressing.MedianHash(pred_example)

        self.status.emit(f"Retrieving files...")
        self.progress.emit(PROGRESS_TRANSFORMED)

        retrieved_indices, distances = addressing.top_k_indices(code_example, pool=codes_search_target,
                                                                k=self.num_results)

        retrieved_paths = [search_paths[i] for i in retrieved_indices]
        retrieved_distances = [distances[i] for i in retrieved_indices]
        binary_codes = [codes_search_target[i].code.__str__() for i in retrieved_indices]

        self.files_retrieved.emit(retrieved_paths, retrieved_distances, binary_codes)

        self.status.emit(f"Finished")
        self.progress.emit(PROGRESS_FINISHED)
        self.finished.emit()

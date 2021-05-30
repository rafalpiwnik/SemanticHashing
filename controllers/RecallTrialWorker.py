import math

import h5py
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

import vdsh.utility
from addressing import medhash_transform
from addressing.metrics import precision, top_k_indices
from controllers.usersetup import load_config


class RecallTrialWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    precisionResult = pyqtSignal(float)
    status = pyqtSignal(str)

    def __init__(self, dataset_name: str, model_name: str):
        super().__init__()

        self._dataset_name = dataset_name
        self._model_name = model_name

    def run(self):
        self.status.emit(f"Loading model '{self._model_name}'...")
        self.progress.emit(5)

        model, _ = vdsh.utility.load_model(self._model_name)

        self.status.emit(f"Loading data...")
        self.progress.emit(10)

        data_home = load_config()["model"]["data_home"]
        try:
            with h5py.File(f"{data_home}/{self._dataset_name}/data.hdf5", "r") as hf:
                train: np.ndarray = hf["train"][:]
                train_targets: np.ndarray = hf["train_labels"][:]
                test: np.ndarray = hf["test"][:]
                test_targets: np.ndarray = hf["test_labels"][:]

                self.status.emit(f"Running predict...")
                self.progress.emit(27)

                train_pred = model.predict(train)
                test_pred = model.predict(test)

                self.status.emit(f"Transforming to binary codes")
                self.progress.emit(42)

                train_codes = medhash_transform(train_pred)
                test_codes = medhash_transform(test_pred)

                current_progress = 42
                end_progress = 95
                steps = len(test_codes)
                progress_per_step = (end_progress - current_progress) / steps

                self.status.emit(f"Running metrics tests...")

                precision_scores = []
                for idx, tc in enumerate(test_codes):
                    r = precision(test_targets[idx], train_targets, top_k_indices(tc, train_codes, 100)[0], 100)
                    precision_scores.append(r)

                    current_progress += progress_per_step
                    self.progress.emit(math.floor(current_progress))

                mean_precision = np.array(precision_scores).mean()
                self.precisionResult.emit(mean_precision)

                self.progress.emit(100)
                self.finished.emit()
                self.status.emit("Finished")
        except (IOError, OSError):
            print(f"Cannot reach data.hdf5 in {self._dataset_name}")

            self.progress.emit(0)
            self.finished.emit()
            self.status.emit("Failed to read data")





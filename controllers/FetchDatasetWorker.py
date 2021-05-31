from datetime import datetime

from PyQt5.QtCore import QObject, pyqtSignal

from storage.datasets import create_20ng


class FetchDatasetWorker(QObject):
    finished = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, dataset_name: str, output_name: str, vocab_size: int):
        """Creates a worker for fetching a dataset from an internet resource

        Parameters
        ----------
        dataset_name : str
            Qualified dataset name from the list of available datasets
        output_name : str
            Output name of the dataset
        vocab_size : int
            Vocabulary size to fetch
        """
        super().__init__()
        self._dataset_name = dataset_name
        self._output_name = output_name
        self._vocab_size = vocab_size

    def run(self):
        """Fetches one of the available datasets or raises NotImplementedError if no such datasets is available"""
        start = datetime.now()

        if self._dataset_name == "20ng":
            self.status.emit("Fetching 20ng... Please be patient")
            create_20ng(self._vocab_size, name=self._output_name)
        else:
            raise NotImplementedError("No such dataset")

        end = datetime.now()

        self.status.emit(f"Finished ({(end - start).seconds}s)")
        self.finished.emit()

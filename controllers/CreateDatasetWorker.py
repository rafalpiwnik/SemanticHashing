import os
from datetime import datetime

import h5py
from PyQt5.QtCore import QObject, pyqtSignal

import storage
from controllers.usersetup import load_config
from storage import DocumentVectorizer
from storage.MetaInfo import DatasetMetaInfo


class CreateDatasetWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, root_dir: str, name: str, vocab_size: int, file_ext=""):
        super().__init__()
        self.root_dir = root_dir
        self.name = name
        self.vocab_size = vocab_size
        self.file_ext = file_ext

    def run(self):
        start = datetime.now()

        data_home = load_config()["model"]["data_home"]
        dest = f"{data_home}/{self.name}"

        self.progress.emit(10)

        try:
            os.mkdir(f"{dest}")
        except FileExistsError:
            pass

        self.status.emit(f"Getting filepaths from: f{self.root_dir}")

        paths = storage.get_paths(self.root_dir, filter_extension=self.file_ext)
        print(f"Found {len(paths)} suitable files at {self.root_dir}")

        self.progress.emit(35)
        self.status.emit(f"Found {len(paths)} files. Vectorizing...")

        print("Vectorizing...")
        v = DocumentVectorizer(self.vocab_size)
        X, words = v.fit_transform(paths)

        self.progress.emit(90)
        self.status.emit(f"Saving dataset to {dest}")

        print("Saving dataset...")
        with h5py.File(f"{dest}/data.hdf5", "w") as hf:
            hf.create_dataset(name="train", data=X.toarray(), compression="gzip")

        storage.save_vectorizer(v, dirpath=f"{dest}")

        self.progress.emit(95)

        mi = DatasetMetaInfo(self.name,
                             self.vocab_size,
                             num_train=X.shape[0],
                             num_test=0,
                             num_labels=0,
                             user=True,
                             source_dir=self.root_dir)

        mi.dump(dest)

        end = datetime.now()

        self.progress.emit(100)
        self.status.emit(f"Finished ({(end - start).seconds}s)")
        self.finished.emit()

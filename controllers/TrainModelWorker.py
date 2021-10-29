import os
import shutil

import tensorflow as tf
from PyQt5.QtCore import pyqtSignal, QObject

import vdsh.utility
from controllers.GuiCallback import GuiCallback
from controllers.usersetup import load_config
from storage.datasets import extract_train

TRAINING_START_MSG = "Starting training..."

MODEL_SAVED_MSG = "Model saved"


# noinspection PyUnresolvedReferences
class TrainModelWorker(QObject):
    """Worker used to train a VDSH model model"""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, model_name: str, dataset_name: str, epochs: int, batch_size: int, optimizer: str,
                 initialRate: float, decaySteps: int,
                 decayRate: float):
        """Create a VDSH train model worker"""
        super().__init__()

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.optimizer = optimizer

        self.epochs = epochs
        self.batch_size = batch_size

        self.decay_steps = decaySteps
        self.decay_rate = decayRate
        self.initial_rate = initialRate

        self.progbar_callback = GuiCallback()

    @property
    def progressbar_callback(self):
        return self.progbar_callback

    def run(self):
        self.status.emit(f"Loading model '{self.model_name}'")

        # Load model and vectorizer
        model, vectorizer = vdsh.utility.load_model(self.model_name)

        # Reset the mode before refitting if it has already been fit
        if model.meta.is_fit:
            model_home = load_config()["model"]["model_home"]
            source = f"{model_home}/{self.model_name}"
            dest = f"{model_home}/{self.model_name}__swap"
            os.rename(source, dest)
            os.mkdir(source)

            model.meta.info["fit"] = False
            model.meta.info["fit_dataset"] = ""
            model.meta.info["fit_time"] = ""
            model.meta.dump(source)

            shutil.rmtree(dest)

            model, vectorizer = vdsh.utility.load_model(self.model_name)

        self.status.emit(f"Model loaded. Extracting train from '{self.dataset_name}' dataset")

        # Extract train dataset
        X = extract_train(self.dataset_name)

        self.status.emit("Compiling the model...")

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)

        if self.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            raise NotImplementedError("No such optimizer")

        model.compile(optimizer=opt)

        self.status.emit(TRAINING_START_MSG)

        model.fit(X, epochs=self.epochs, batch_size=self.batch_size, callbacks=[self.progbar_callback])

        # Flag model fit
        model.meta.flag_fit(self.dataset_name)

        # Fitted model is saved, and marked fit, it cannot be fit once more without copying
        vdsh.utility.dump_model(model)

        self.status.emit(MODEL_SAVED_MSG)
        self.finished.emit()

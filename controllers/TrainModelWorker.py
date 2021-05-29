from PyQt5.QtCore import pyqtSignal, QObject

import vdsh.utility
from controllers.usersetup import load_config
from storage.datasets import extract_train
import tensorflow as tf


class TrainModelWorker(QObject):
    finished = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, model_name: str, dataset_name: str, epochs: int, batch_size: int, optimizer: str,
                 initialRate: float, decaySteps: int,
                 decayRate: float):
        super().__init__()

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.optimizer = optimizer

        self.epochs = epochs
        self.batch_size = batch_size

        self.decaySteps = decaySteps
        self.decayRate = decayRate
        self.initialRate = initialRate

    def run(self):
        self.status.emit(f"Loading model '{self.model_name}'")

        # Load model and vectorizer
        model, vectorizer = vdsh.utility.load_model(self.model_name)

        self.status.emit(f"Model loaded. Extracting train from '{self.dataset_name}' dataset")

        # Create callbacks for monitoring metrics
        checkpoint_filepath = load_config()["model"]["model_home"] + "/checkpoint"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor="val_loss",
            mode="max",
            save_best_only=True)

        # Extract train dataset
        X = extract_train(self.dataset_name)

        self.status.emit("Compiling the model...")

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initialRate,
            decay_steps=self.decaySteps,
            decay_rate=self.decayRate,
            staircase=True)

        if self.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            raise NotImplementedError("No such optimizer")

        model.compile(optimizer=opt)

        self.status.emit(f"Starting training...")

        model.fit(X, epochs=self.epochs, batch_size=self.batch_size, callbacks=[model_checkpoint_callback])

        # Fitted model is saved, and marked fit, it cannot be fit once more without copying
        vdsh.utility.dump_model(model)

        self.status.emit("Model saved")
        self.finished.emit()

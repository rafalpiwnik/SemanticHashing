import math

from PyQt5.QtCore import pyqtSignal, QObject
from tensorflow.python.keras.callbacks import Callback


# noinspection PyUnresolvedReferences
class GuiCallback(QObject, Callback):
    learningProgress = pyqtSignal(int)
    epochProgress = pyqtSignal(int)
    metrics = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.epoch_step = 0
        self.epoch_num = 0

    def on_epoch_begin(self, epoch, logs=None):
        """On epoch begin update the total progress and emit it"""
        self.epoch_step = 0
        self.epoch_num += 1

        total_progress = min(math.floor(100 * (self.epoch_num / self.params["epochs"])), 100)
        self.learningProgress.emit(total_progress)

    def on_batch_end(self, batch, logs=None):
        """On batch end update the total progress and emi it"""
        self.epoch_step += 1
        current_progress = min(math.floor(100 * (self.epoch_step / self.params["steps"])), 100)

        self.epochProgress.emit(current_progress)
        self.metrics.emit(self.metrics_message(logs))

    def on_train_begin(self, logs=None):
        self.metrics.emit("Starting training...")

    def on_train_end(self, logs=None):
        self.metrics.emit("Training finished. Saving weights...")

    @staticmethod
    def metrics_message(logs=None):
        if logs:
            return f"Total loss: {logs['loss']:.2f}, reconstruction_loss: {logs['reconstruction_loss']:.2f}," \
                   f" effective kl_loss: {logs['kl_loss']:.2f}"
        else:
            return "No metrics"

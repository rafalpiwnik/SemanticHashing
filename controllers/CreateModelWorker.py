import os

from PyQt5.QtCore import QObject, pyqtSignal

import vdsh.utility
from controllers.usersetup import load_config


class CreateModelWorker(QObject):
    finished = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, vocab_dim: int, hidden_dim: int, latent_dim: int, kl_step: float, dropout_prob: float,
                 name: str = "default"):
        super().__init__()

        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kl_step = kl_step
        self.dropout_prob = dropout_prob
        self.name = name

    def run(self):
        self.status.emit(f"Creating VDSH '{self.name}'")

        mock_mi = vdsh.utility.create_mock_model(self.vocab_dim, self.hidden_dim, self.latent_dim, self.kl_step,
                                                 self.dropout_prob, self.name)

        self.status.emit("Model saved")
        self.finished.emit()

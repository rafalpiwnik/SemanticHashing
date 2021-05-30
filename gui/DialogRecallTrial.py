from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread

from controllers.RecallTrialWorker import RecallTrialWorker
from gui.designer.Ui_TestDialog import Ui_TestDialog


class DialogRecallTrial(QtWidgets.QDialog, Ui_TestDialog):
    progress = pyqtSignal(int)

    def __init__(self, dataset_name, model_name, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.thread = QThread()
        self.worker = RecallTrialWorker(dataset_name, model_name)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Signals
        self.worker.status.connect(self.update_status)
        self.worker.progress.connect(self.update_progbar)
        self.worker.precisionResult.connect(self.set_precision_result)

        self.thread.start()

        # Buttons
        self.finishTestButton.clicked.connect(self.exit_on_finished_clicked)

    @pyqtSlot()
    def exit_on_finished_clicked(self):
        self.close()

    @pyqtSlot(str)
    def update_status(self, text: str):
        self.statusMessage.setText(text)

    @pyqtSlot(int)
    def update_progbar(self, value: int):
        self.progressBar.setValue(value)

    @pyqtSlot(float)
    def set_precision_result(self, value: float):
        self.precision.setText(f"{value:.2}")


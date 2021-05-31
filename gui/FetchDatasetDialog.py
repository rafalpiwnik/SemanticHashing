from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog

from controllers.FetchDatasetWorker import FetchDatasetWorker
from gui.designer.Ui_FetchDatasetDialog import Ui_FetchDatasetDialog


class FetchDatasetDialog(QDialog, Ui_FetchDatasetDialog):
    datasetsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.setWindowIcon(QIcon("../resources/dataset-fetch.png"))

        self.createDatasetButton.clicked.connect(self.start_fetching)
        self.closeButton.clicked.connect(self.exit_wizard)

    @pyqtSlot(str)
    def update_status(self, text: str):
        self.statusMessage.setText(text)

    @pyqtSlot()
    def start_fetching(self):
        self.datasetName.setDisabled(True)
        self.outputName.setDisabled(True)
        self.vocabSize.setDisabled(True)
        self.stopwordsChoice.setDisabled(True)
        self.createDatasetButton.setDisabled(True)

        self.thread = QThread()
        self.worker = FetchDatasetWorker(self.datasetName.currentText(), self.outputName.text(), self.vocabSize.value())
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Signals
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(lambda: self.datasetsChanged.emit())
        self.worker.finished.connect(lambda: self.closeButton.setEnabled(True))

        self.thread.start()

    def exit_wizard(self):
        self.close()

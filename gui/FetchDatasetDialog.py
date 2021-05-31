from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog

from controllers.FetchDatasetWorker import FetchDatasetWorker
from controllers.controller import check_dataset_available
from gui.designer.Ui_FetchDatasetDialog import Ui_FetchDatasetDialog


class FetchDatasetDialog(QDialog, Ui_FetchDatasetDialog):
    datasetsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.setWindowIcon(QIcon("../resources/dataset-fetch.png"))
        self.infoStyle = 'QLabel { color: black; font: "Segoe UI"; font-size: 13px }'
        self.warningStyle = 'QLabel { color: black; font: italic "Segoe UI"; font-size: 13px }'

        self.fetchDatasetButton.clicked.connect(self.start_fetching)
        self.closeButton.clicked.connect(self.exit_wizard)

        self.outputName.textChanged.connect(self.check_dataset_overwrite)
        self.outputName.textChanged.connect(self.enable_button_on_name_filled)

        self.check_dataset_overwrite(self.outputName.text())

    @pyqtSlot(str)
    def check_dataset_overwrite(self, output_name: str):
        meta_info = check_dataset_available(output_name)
        if meta_info and self.outputName.text() != "":
            self.statusMessage.setStyleSheet(self.warningStyle)
            self.statusMessage.setText(f"Fetching the dataset will overwrite the previous one."
                                       f" Change name if you wish to avoid this!")
        else:
            self.statusMessage.setStyleSheet(self.infoStyle)
            self.statusMessage.setText("Waiting for start...")

    @pyqtSlot(str)
    def enable_button_on_name_filled(self, name: str):
        if self.outputName.text() == "":
            self.fetchDatasetButton.setDisabled(True)
        else:
            self.fetchDatasetButton.setEnabled(True)

    @pyqtSlot(str)
    def update_status(self, text: str):
        self.statusMessage.setText(text)

    @pyqtSlot()
    def start_fetching(self):
        """Starts the FetchDatasetWorker and disables the worker"""
        self.datasetName.setDisabled(True)
        self.outputName.setDisabled(True)
        self.vocabSize.setDisabled(True)
        self.stopwordsChoice.setDisabled(True)
        self.fetchDatasetButton.setDisabled(True)

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

import os.path

from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl, QThread, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog

from controllers.CreateDatasetWorker import CreateDatasetWorker
from controllers.controller import check_dataset_available
from gui.designer.Ui_DatasetWizard import Ui_DatasetWizardDialog


class DatasetWizard(QtWidgets.QDialog, Ui_DatasetWizardDialog):
    datasetsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super(DatasetWizard, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowIcon(QIcon("../resources/dataset-add.png"))

        self.createDatasetButton.setDisabled(True)
        self.outputName.textChanged.connect(self.enable_button_on_name_filled)

        # DIALOG DIR CHOICE
        self.chooseDirectoryButton.clicked.connect(self.open_dialog_choose_dir)

        # COMMIT
        self.createDatasetButton.clicked.connect(self.create_dataset)

        # CHECKING FOR OVERWRITE
        self.infoStyle = 'QLabel { color: black; font: "Segoe UI"; font-size: 13px }'
        self.warningStyle = 'QLabel { color: black; font: italic "Segoe UI"; font-size: 13px }'

        self.outputName.textChanged.connect(self.check_dataset_overwrite)
        self.outputName.textChanged.connect(self.enable_button_on_name_filled)

    @pyqtSlot(str)
    def check_dataset_overwrite(self, output_name: str):
        meta_info = check_dataset_available(output_name)
        if meta_info and self.outputName.text() != "":
            self.statusMessage.setStyleSheet(self.warningStyle)
            self.statusMessage.setText(f"Creating the dataset will overwrite the previous one."
                                       f" Change name if you wish to avoid this!")
        else:
            self.statusMessage.setStyleSheet(self.infoStyle)
            self.statusMessage.setText("Waiting for start...")

    @pyqtSlot(str)
    def enable_button_on_name_filled(self, text: str):
        if self.outputName.text() == "":
            self.createDatasetButton.setDisabled(True)
        else:
            self.createDatasetButton.setEnabled(True)

    @pyqtSlot()
    def open_dialog_choose_dir(self):
        home = os.path.expanduser("~")
        fileDialog = QFileDialog(directory=home)
        dirUrl: QUrl = fileDialog.getExistingDirectoryUrl()
        dirpath = dirUrl.path()[1:]
        default_name = dirpath.split("/")[-1]

        self.directoryChoiceStatus.setText(dirpath)
        self.outputName.setText(default_name)

    @pyqtSlot(int)
    def update_progbar(self, value: int):
        self.progressBar.setValue(value)

    @pyqtSlot(str)
    def update_status(self, msg: str):
        self.statusMessage.setText(msg)

    @pyqtSlot()
    def exit_on_worker_finished(self):
        self.datasetsChanged.emit()
        self.close()

    @pyqtSlot()
    def create_dataset(self):
        dirpath = self.directoryChoiceStatus.text()
        name = self.outputName.text()

        self.createDatasetButton.setDisabled(True)
        self.outputName.setDisabled(True)
        self.vocabSize.setDisabled(True)
        self.stopwordsChoice.setDisabled(True)
        self.createDatasetButton.setDisabled(True)
        self.chooseDirectoryButton.setDisabled(True)

        self.thread = QThread()
        self.worker = CreateDatasetWorker(dirpath, name, int(self.vocabSize.text()))
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.update_progbar)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.exit_on_worker_finished)

        self.thread.start()

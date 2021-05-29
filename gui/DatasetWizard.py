import os.path

from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl, QThread, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from controllers.CreateDatasetWorker import CreateDatasetWorker
from gui.designer.Ui_DatasetWizard import Ui_DatasetWizardDialog


class DatasetWizard(QtWidgets.QDialog, Ui_DatasetWizardDialog):
    datasetsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super(DatasetWizard, self).__init__(parent=parent)
        self.setupUi(self)

        # parent is the main window

        self.createDatasetButton.setDisabled(True)
        self.outputName.textChanged.connect(self.enableButtonOnNameFilled)

        # DIALOG DIR CHOICE
        self.chooseDirectoryButton.clicked.connect(self.open_dialog_choose_dir)

        # COMMIT
        self.createDatasetButton.clicked.connect(self.create_dataset)

    @pyqtSlot(str)
    def enableButtonOnNameFilled(self, text: str):
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
        self.createWorkerStatus.setText(msg)

    @pyqtSlot()
    def exit_on_worker_finished(self):
        self.datasetsChanged.emit()
        self.close()

    @pyqtSlot()
    def create_dataset(self):
        dirpath = self.directoryChoiceStatus.text()
        name = self.outputName.text()

        self.createDatasetButton.setDisabled(True)

        # controllers.controller.create_user_dataset(dirpath, int(self.vocabSize.text()), name, progbar=self.progbar)
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

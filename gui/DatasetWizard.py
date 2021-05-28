import os.path

from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl, QThread
from PyQt5.QtWidgets import QFileDialog

import controllers.controller
from gui.designer.Ui_DatasetWizard import Ui_DatasetWizardDialog


class DatasetWizard(QtWidgets.QDialog, Ui_DatasetWizardDialog):
    def __init__(self, parent=None):
        super(DatasetWizard, self).__init__(parent=parent)
        self.setupUi(self)

        # parent is the main window

        # DIALOG DIR CHOICE
        self.chooseDirectoryButton.clicked.connect(self.open_dialog_choose_dir)

        # COMMIT
        self.createDatasetButton.clicked.connect(self.create_dataset)

        # ADDitions
        self.progbar = QtWidgets.QProgressBar()
        self.progbar.setMinimum(0)
        self.progbar.setMaximum(100)
        self.expandingGridLayout.addWidget(self.progbar)

    def open_dialog_choose_dir(self):
        home = os.path.expanduser("~")
        fileDialog = QFileDialog(directory=home)
        dirUrl: QUrl = fileDialog.getExistingDirectoryUrl()
        dirpath = dirUrl.path()[1:]
        default_name = dirpath.split("/")[-1]

        self.directoryChoiceStatus.setText(dirpath)
        self.outputName.setText(default_name)

    def create_dataset(self):
        dirpath = self.directoryChoiceStatus.text()
        name = self.outputName.text()

        controllers.controller.create_user_dataset(dirpath, int(self.vocabSize.text()), name, progbar=self.progbar)

        if self.parent():
            self.parent().update_datasets()

        self.close()

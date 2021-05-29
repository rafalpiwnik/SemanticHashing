import os
import sys

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QEvent, Qt, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog

from controllers import controller
from controllers.controller import fetch_datasets_to_widgets, fetch_models_to_widgets
from gui.DatasetWidget import DatasetWidget
from gui.DatasetWizard import DatasetWizard
from gui.ModelWidget import ModelWidget
from gui.ModelWizard import ModelWizard
from gui.TrainWizard import TrainWizard
from gui.designer.Ui_MainWindow import Ui_MainWindow
from storage.MetaInfo import are_compatible


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

        # DATASET STACKS
        self.datasetStacks = [self.trainDatasetStack, self.testDatasetStack]
        self.modelStacks = [self.trainModelStack, self.testModelStack, self.searchModelStack]

        # DATASET WIDGET STACKS SETUP
        for stack in self.datasetStacks:
            dw = DatasetWidget()
            dw.make_prompt_preset()
            stack.addWidget(dw)
            stack.setCurrentWidget(dw)

        # MODEL WIDGET STACKS SETUP
        for stack in self.modelStacks:
            mw = ModelWidget()
            mw.make_prompt_preset()
            stack.addWidget(mw)
            stack.setCurrentWidget(mw)

        # STACK MIXING COMPATIBLE SIGNAL
        for stack in self.datasetStacks + self.modelStacks:
            stack.currentChanged.connect(self.check_mixing_compatible)

        # POPULATE WITH ENTITIES
        self.update_datasets()
        self.update_models()

        # TOOLBAR ACTIONS
        self.actionNew_dataset.triggered.connect(self.open_dataset_wizard)
        self.actionNew_model.triggered.connect(self.open_model_wizard)

        # TRAIN BUTTONS
        self.buttonTrainWizard.clicked.connect(self.open_train_wizard)

        # DATASETS LIST
        self.datasetList.itemDoubleClicked.connect(self.update_current_dataset)

        # MODELS LIST
        self.modelList.itemDoubleClicked.connect(self.update_current_model)

        # SEARCH TAB
        self.dirIcon.setPixmap(QPixmap("../resources/dir-inactive.png"))
        self.fileIcon.setPixmap(QPixmap("../resources/file-inactive.png"))

        self.chooseDirButton.clicked.connect(self.open_dialog_choose_search_dir)
        self.chooseFileButton.clicked.connect(self.open_dialog_choose_search_file)

        self.searchModelStack.currentChanged.connect(self.check_search_available)

    @pyqtSlot(QtWidgets.QListWidgetItem)
    def update_current_dataset(self, item: QtWidgets.QListWidgetItem):
        """Changes currently displayed item after double click on it"""
        list_widget: DatasetWidget = self.datasetList.itemWidget(item)
        for stack in self.datasetStacks:
            cloned_widget = list_widget.clone()
            cloned_widget.setContextMenuPolicy(Qt.PreventContextMenu)

            old_widget = stack.currentWidget()
            stack.addWidget(cloned_widget)
            stack.setCurrentWidget(cloned_widget)
            stack.removeWidget(old_widget)

    def verify_current_dataset(self):
        for stack in self.datasetStacks:
            current: DatasetWidget = stack.currentWidget()
            meta_info = controller.check_dataset_available(current.name.text())
            if not meta_info:
                dw = DatasetWidget()
                dw.make_prompt_preset()
                dw.setContextMenuPolicy(Qt.PreventContextMenu)
                stack.addWidget(dw)
                stack.setCurrentWidget(dw)
                stack.removeWidget(current)
            else:
                current.set_fields(meta_info)
                self.check_mixing_compatible()

    @pyqtSlot(QtWidgets.QListWidgetItem)
    def update_current_model(self, item: QtWidgets.QListWidgetItem):
        list_widget: ModelWidget = self.modelList.itemWidget(item)
        for stack in self.modelStacks:
            cloned_widget = list_widget.clone()
            cloned_widget.setContextMenuPolicy(Qt.PreventContextMenu)

            old_widget = stack.currentWidget()
            stack.addWidget(cloned_widget)
            stack.setCurrentWidget(cloned_widget)
            stack.removeWidget(old_widget)

    def verify_current_model(self):
        """Verifies if the current model in any of the stacks is still available locally"""
        for stack in self.modelStacks:
            current: ModelWidget = stack.currentWidget()
            model_mi = controller.check_model_available(current.name.text())
            if not model_mi:
                mw = ModelWidget()
                mw.make_prompt_preset()
                mw.setContextMenuPolicy(Qt.PreventContextMenu)
                stack.addWidget(mw)
                stack.setCurrentWidget(mw)
                stack.removeWidget(current)
            else:
                current.set_fields(model_mi)
                self.check_mixing_compatible()

    def check_mixing_compatible(self):
        for ds, ms in zip(self.datasetStacks, self.modelStacks):
            dataset: DatasetWidget = ds.currentWidget()
            model: ModelWidget = ms.currentWidget()

            if model and dataset:

                model.reset_state()
                dataset.reset_state()

                if (
                        not dataset.vocabulary.text() == model.vocab.text()) and not dataset.is_prompt and not model.is_prompt:
                    dataset.mark_mismatch_error()
                    model.mark_mismatch_error()
                elif dataset.name.text() == model.fit.text():
                    dataset.mark_native()
                    model.mark_native()

                if model.is_prompt:
                    model.set_inactive_icon()
                    self.buttonTrainWizard.setEnabled(False)
                if dataset.is_prompt:
                    dataset.set_inactive_icon()
                    self.buttonTrainWizard.setEnabled(False)

                if not model.is_prompt and not dataset.is_prompt:
                    if dataset.vocabulary.text() == model.vocab.text():
                        self.buttonTrainWizard.setEnabled(True)
                        pass
                    else:
                        self.buttonTrainWizard.setEnabled(False)
                        pass

    @pyqtSlot()
    def open_dataset_wizard(self):
        dialog = DatasetWizard(parent=self)
        dialog.datasetsChanged.connect(self.update_datasets)
        dialog.exec_()

    @pyqtSlot()
    def open_train_wizard(self):
        dialog = TrainWizard(self.trainDatasetStack.currentWidget(), self.trainModelStack.currentWidget(),
                             parent=self)
        dialog.modelsChanged.connect(self.update_models)
        dialog.exec_()

    @pyqtSlot()
    def open_model_wizard(self):
        dialog = ModelWizard(self.datasetList)
        dialog.modelsChanged.connect(self.update_models)
        dialog.exec_()

    @pyqtSlot()
    def update_datasets(self):
        self.verify_current_dataset()

        widgets = fetch_datasets_to_widgets()

        self.datasetList.clear()

        for w in widgets:
            w.datasetRemoved.connect(self.update_datasets)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(w.sizeHint())
            self.datasetList.addItem(item)
            self.datasetList.setItemWidget(item, w)

    @pyqtSlot()
    def update_models(self):
        self.verify_current_model()

        widgets = fetch_models_to_widgets()

        self.modelList.clear()

        for w in widgets:
            w.modelRemoved.connect(self.update_models)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(w.sizeHint())
            self.modelList.addItem(item)
            self.modelList.setItemWidget(item, w)

    @pyqtSlot()
    def check_search_available(self):
        model_widget: ModelWidget = self.searchModelStack.currentWidget()
        if not model_widget.is_prompt and self.dirname != "" and self.filename != "":
            self.runSearchButton.setEnabled(True)
        else:
            self.runSearchButton.setEnabled(False)

    @pyqtSlot()
    def open_dialog_choose_search_dir(self):
        home = os.path.expanduser("~")
        fileDialog = QFileDialog(directory=home)
        dirUrl: QUrl = fileDialog.getExistingDirectoryUrl()
        dirpath = dirUrl.path()[1:]
        self.dirname.setText(dirpath)
        self.dirIcon.setPixmap(QPixmap("../resources/dir.png"))
        self.check_search_available()

    @pyqtSlot()
    def open_dialog_choose_search_file(self):
        home = os.path.expanduser("~")
        fileDialog = QFileDialog(directory=home)
        fileUrl: QUrl = fileDialog.getOpenFileUrl()[0]
        path = fileUrl.path()[1:]
        self.filename.setText(path)
        self.fileIcon.setPixmap(QPixmap("../resources/file.png"))
        self.check_search_available()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

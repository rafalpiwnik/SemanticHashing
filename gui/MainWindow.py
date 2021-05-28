import sys

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QEvent, Qt

from controllers import controller
from controllers.controller import fetch_datasets_to_widgets, fetch_models_to_widgets
from gui.DatasetWidget import DatasetWidget
from gui.DatasetWizard import DatasetWizard
from gui.ModelWidget import ModelWidget
from gui.designer.Ui_MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

        # DATASET STACKS
        self.datasetStacks = [self.trainDatasetStack]
        self.modelStacks = [self.trainModelStack]

        # DATASET WIDGET TRAIN STACK
        dw = DatasetWidget()
        dw.make_prompt_preset()
        dw.setContextMenuPolicy(Qt.PreventContextMenu)
        self.trainDatasetStack.addWidget(dw)
        self.trainDatasetStack.setCurrentWidget(dw)

        mw = ModelWidget()

        self.trainModelStack.addWidget(mw)
        self.trainModelStack.setCurrentWidget(mw)

        # TOOLBAR ACTIONS
        self.actionNew_dataset.triggered.connect(self.open_dataset_wizard)

        # DATASETS LIST
        self.datasetList.itemDoubleClicked.connect(self.update_current_dataset)

    @pyqtSlot(QtWidgets.QListWidgetItem)
    def update_current_dataset(self, item: QtWidgets.QListWidgetItem):
        """Changes currently displayed item after double click on it"""
        list_widget: DatasetWidget = self.datasetList.itemWidget(item)
        cloned_widget = list_widget.clone()
        cloned_widget.setContextMenuPolicy(Qt.PreventContextMenu)

        for stack in self.datasetStacks:
            old_widget = stack.currentWidget()
            stack.addWidget(cloned_widget)
            stack.setCurrentWidget(cloned_widget)
            stack.removeWidget(old_widget)

    def verify_current_dataset(self):
        for stack in self.datasetStacks:
            current: DatasetWidget = stack.currentWidget()
            exists = controller.check_dataset_available(current.name.text())
            if not exists:
                dw = DatasetWidget()
                dw.make_prompt_preset()
                dw.setContextMenuPolicy(Qt.PreventContextMenu)
                stack.addWidget(dw)
                stack.setCurrentWidget(dw)
                stack.removeWidget(current)

    @pyqtSlot(QtWidgets.QListWidgetItem)
    def update_current_model(self, item: QtWidgets.QListWidgetItem):
        list_widget: ModelWidget = self.modelList.itemWidget(item)
        cloned_widget = list_widget.clone()
        cloned_widget.setContextMenuPolicy(Qt.PreventContextMenu)

        for stack in self.modelStacks:
            old_widget = stack.currentWidget()
            stack.addWidget(cloned_widget)
            stack.setCurrentWidget(cloned_widget)
            stack.removeWidget(old_widget)

    def verify_current_model(self):
        for stack in self.modelStacks:
            current: ModelWidget = stack.currentWidget()
            exists = controller.check_dataset_available(current.name.text())
            if not exists:
                mw = ModelWidget()
                mw.make_prompt_preset()
                mw.setContextMenuPolicy(Qt.PreventContextMenu)
                stack.addWidget(mw)
                stack.setCurrentWidget(mw)
                stack.removeWidget(current)

    @pyqtSlot()
    def open_dataset_wizard(self):
        dialog = DatasetWizard(parent=self)
        dialog.datasetsChanged.connect(self.update_datasets)
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

    def set_models(self, widgets: list[ModelWidget]):
        for w in widgets:
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(w.sizeHint())
            self.modelList.addItem(item)
            self.modelList.setItemWidget(item, w)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    window.update_datasets()
    window.set_models(fetch_models_to_widgets())

    sys.exit(app.exec_())

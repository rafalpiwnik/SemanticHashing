import sys

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QEvent

from controllers.controller import fetch_datasets_to_widgets, fetch_models_to_widgets
from gui.DatasetWidget import DatasetWidget
from gui.DatasetWizard import DatasetWizard
from gui.ModelWidget import ModelWidget
from gui.designer.Ui_MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

        # self.trainMixingGridLayout.addWidget(ModelWidget(), 0, 1)
        dw = DatasetWidget()
        # dw.mark_mismatch_error()

        self.trainDatasetStack.addWidget(dw)
        self.trainDatasetStack.setCurrentWidget(dw)

        mw = ModelWidget()
        # mw.mark_mismatch_error()

        self.trainModelStack.addWidget(mw)
        self.trainModelStack.setCurrentWidget(mw)

        # TOOLBAR ACTIONS
        self.actionNew_dataset.triggered.connect(self.open_dataset_wizard)

        # DATASETS LIST
        self.datasetList.itemDoubleClicked.connect(self.update_current_dataset)

    def update_current_dataset(self, item: QtWidgets.QListWidgetItem):
        """Slot, changes currently displayed item"""
        list_widget: DatasetWidget = self.datasetList.itemWidget(item)
        cloned_widget = list_widget.clone()

        old_widget = self.trainDatasetStack.currentWidget()
        self.trainDatasetStack.addWidget(cloned_widget)
        self.trainDatasetStack.setCurrentWidget(cloned_widget)
        self.trainDatasetStack.removeWidget(old_widget)

    @pyqtSlot()
    def open_dataset_wizard(self):
        dialog = DatasetWizard(parent=self)
        dialog.datasetsChanged.connect(self.update_datasets)
        dialog.exec_()

    @pyqtSlot()
    def update_datasets(self):
        widgets = fetch_datasets_to_widgets()

        self.datasetList.clear()

        for w in widgets:
            w.mw = self
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(w.sizeHint())
            self.datasetList.addItem(item)
            self.datasetList.setItemWidget(item, w)

        # TODO this has to check if dataset in the mixing table exitsts as well

    def set_models(self, widgets: list[ModelWidget]):
        for w in widgets:
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(w.sizeHint())
            self.modelList.addItem(item)
            self.modelList.setItemWidget(item, w)

    """
    def eventFilter(self, source, event):
        if event.type() == QEvent.ContextMenu and source is self.datasetList:
            menu = QtWidgets.QMenu()
            menu.addAction('Open Window')
            if menu.exec_(event.globalPos()):
                item = source.itemAt(event.pos())
                print(item.text())
            return True
        return True
    """


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    window.update_datasets()
    window.set_models(fetch_models_to_widgets())

    sys.exit(app.exec_())

import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import pyqtSlot

from controllers.controller import fetch_datasets_to_widgets, fetch_models_to_widgets
from gui.DatasetWidget import DatasetWidget
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

        # SIGNALS / SLOTS
        self.datasetList.itemDoubleClicked.connect(self.update_dataset)

    def update_dataset(self, item: QtWidgets.QListWidgetItem):
        """Slot, changes currently displayed item"""
        list_widget: DatasetWidget = self.datasetList.itemWidget(item)
        cloned_widget = list_widget.clone()

        old_widget = self.trainDatasetStack.currentWidget()
        self.trainDatasetStack.addWidget(cloned_widget)
        self.trainDatasetStack.setCurrentWidget(cloned_widget)
        self.trainDatasetStack.removeWidget(old_widget)

    def set_datasets(self, widgets: list[DatasetWidget]):
        for w in widgets:
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

    window.set_datasets(fetch_datasets_to_widgets())
    window.set_models(fetch_models_to_widgets())

    sys.exit(app.exec_())

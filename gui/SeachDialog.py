from PyQt5 import QtWidgets

from gui.designer.Ui_SearchDialog import Ui_SearchDialog


class SearchDialog(QtWidgets.QDialog, Ui_SearchDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

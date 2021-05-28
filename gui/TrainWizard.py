from PyQt5 import QtWidgets

from gui.DatasetWidget import DatasetWidget
from gui.ModelWidget import ModelWidget
from gui.designer.Ui_TrainWizard import Ui_TrainWizard


class TrainWizard(QtWidgets.QDialog, Ui_TrainWizard):
    def __init__(self, dw: DatasetWidget, mw: ModelWidget, parent=None):
        super(TrainWizard, self).__init__(parent=parent)
        self.setupUi(self)

        self.datasetWidget = dw.clone()
        self.modelWidget = mw.clone()

        self.datasetSpace.addWidget(self.datasetWidget)
        self.modelSpace.addWidget(self.modelWidget)



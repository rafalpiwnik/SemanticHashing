from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal

from controllers.TrainModelWorker import TrainModelWorker
from gui.DatasetWidget import DatasetWidget
from gui.ModelWidget import ModelWidget
from gui.designer.Ui_TrainWizard import Ui_TrainWizard


class TrainWizard(QtWidgets.QDialog, Ui_TrainWizard):
    modelsChanged = pyqtSignal()

    def __init__(self, dw: DatasetWidget, mw: ModelWidget, parent=None):
        super(TrainWizard, self).__init__(parent=parent)
        self.setupUi(self)

        self.datasetWidget = dw.clone()
        self.modelWidget = mw.clone()

        self.datasetSpace.addWidget(self.datasetWidget)
        self.modelSpace.addWidget(self.modelWidget)

        # Connections
        self.fitModelButton.clicked.connect(self.startTraining)

    @pyqtSlot(str)
    def update_status(self, text: str):
        self.statusMessage.setText(text)

    @pyqtSlot()
    def exit_on_worker_finished(self):
        self.modelsChanged.emit()
        self.close()

    @pyqtSlot()
    def startTraining(self):
        dataset_name = self.datasetWidget.name.text()
        model_name = self.modelWidget.name.text()
        vocab_size = int(self.modelWidget.vocab.text().replace(",", ""))
        hidden_dim = int(self.modelWidget.hiddenDim.text().replace(",", ""))
        latent_dim = int(self.modelWidget.latentDim.text())

        epochs = self.trainEpochs.value()
        batch_size = self.trainBatch.value()
        opt_name = self.optimizer.currentText()

        initial_rate = self.initialRate.value()
        decaySteps = self.decaySteps.value()
        decayRate = self.decayRate.value()

        self.thread = QThread()
        self.worker = TrainModelWorker(model_name,
                                       dataset_name,
                                       epochs,
                                       batch_size,
                                       opt_name,
                                       initial_rate,
                                       decaySteps,
                                       decayRate)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.exit_on_worker_finished)

        self.thread.start()

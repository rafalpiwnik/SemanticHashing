from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal

from controllers.TrainModelWorker import TrainModelWorker
from gui.DatasetWidget import DatasetWidget
from gui.ModelWidget import ModelWidget
from gui.designer.Ui_TrainWizard import Ui_TrainWizard


class TrainWizard(QtWidgets.QDialog, Ui_TrainWizard):
    modelsChanged = pyqtSignal()

    def __init__(self, dw: DatasetWidget, mw: ModelWidget, parent=None):
        """Creates a training wizard, which can run the training process"""
        super(TrainWizard, self).__init__(parent=parent)
        self.setupUi(self)

        self.datasetWidget = dw.clone()
        self.modelWidget = mw.clone()

        self.datasetSpace.addWidget(self.datasetWidget)
        self.modelSpace.addWidget(self.modelWidget)

        # Connections
        self.fitModelButton.clicked.connect(self.startTraining)

        self.worker = None
        self.thread = None

    def make_input_disabled(self):
        elements = [self.optimizer, self.trainEpochs, self.trainBatch, self.initialRate, self.decaySteps,
                    self.decayRate]
        for e in elements:
            e.setDisabled(True)

    @pyqtSlot(str)
    def update_status(self, text: str):
        self.statusMessage.setText(text)

    @pyqtSlot()
    def exit_on_worker_finished(self):
        """Close the window and emit modelsChanged to update models displayed in the main window"""
        self.modelsChanged.emit()
        self.close()

    @pyqtSlot(int)
    def update_epoch_progbar(self, value: int):
        self.epochProgbar.setValue(value)

    @pyqtSlot(int)
    def update_learning_progbar(self, value: int):
        self.learningProgbar.setValue(value)

    @pyqtSlot()
    def startTraining(self):
        dataset_name = self.datasetWidget.name.text()
        model_name = self.modelWidget.name.text()

        self.fitModelButton.setDisabled(True)
        self.make_input_disabled()

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

        # Progressbar callback of the worker
        callback_progbar = self.worker.progressbar_callback
        callback_progbar.learningProgress.connect(self.update_learning_progbar)
        callback_progbar.epochProgress.connect(self.update_epoch_progbar)
        callback_progbar.metrics.connect(self.update_status)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.exit_on_worker_finished)

        self.thread.daemon = True
        self.thread.start()

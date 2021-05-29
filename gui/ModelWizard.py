from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal

import controllers.controller
from controllers.CreateModelWorker import CreateModelWorker
from gui.DatasetWidget import DatasetWidget
from gui.designer.Ui_ModelWizard import Ui_ModelWizard


class ModelWizard(QtWidgets.QDialog, Ui_ModelWizard):
    modelsChanged = pyqtSignal()

    def __init__(self, datasets: QtWidgets.QListWidget, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.createModelButton.setDisabled(True)

        num_items = datasets.count()
        for i in range(num_items):
            item = datasets.item(i)
            widget = datasets.itemWidget(item).clone()

            new_item = QtWidgets.QListWidgetItem()
            new_item.setSizeHint(widget.sizeHint())
            self.datasetsChooseList.addItem(new_item)
            self.datasetsChooseList.setItemWidget(new_item, widget)

        self.infoStyle = 'QLabel { color: black; font: "Segoe UI"; font-size: 13px }'
        self.warningStyle = 'QLabel { color: black; font: italic "Segoe UI"; font-size: 13px }'

        # SIGNAL TO INFER PARAMS
        self.datasetsChooseList.itemDoubleClicked.connect(self.inferVocabSize)

        # CHANGED OUTPUT NAME
        self.outputModelName.textChanged.connect(self.checkModelOverwrite)
        self.outputModelName.textChanged.connect(self.enableButtonOnNameFilled)

        # COMMIT CREATE
        self.createModelButton.clicked.connect(self.onModelCreateClicked)

    @pyqtSlot(QtWidgets.QListWidgetItem)
    def inferVocabSize(self, item: QtWidgets.QListWidgetItem):
        widget: DatasetWidget = self.datasetsChooseList.itemWidget(item)
        vocab_size = int(widget.vocabulary.text().replace(",", ""))
        if self.outputModelName.text() == "":
            self.outputModelName.setText(widget.name.text())
        self.vocabSize.setValue(vocab_size)

    @pyqtSlot(str)
    def checkModelOverwrite(self, output_name: str):
        meta_info = controllers.controller.check_model_available(output_name)
        if meta_info:
            self.statusMessage.setStyleSheet(self.warningStyle)
            self.statusMessage.setText(f"Creating model will overwrite the previous one."
                                       f" Change name if you wish to avoid this!")
        else:
            self.statusMessage.setStyleSheet(self.infoStyle)
            self.statusMessage.setText("Fill in parameters...")

    @pyqtSlot(str)
    def enableButtonOnNameFilled(self, name: str):
        if self.outputModelName.text() == "":
            self.createModelButton.setDisabled(True)
        else:
            self.createModelButton.setEnabled(True)

    @pyqtSlot(str)
    def update_status(self, text: str):
        self.statusMessage.setStyleSheet(self.infoStyle)
        self.statusMessage.setText(text)

    @pyqtSlot()
    def exit_on_worker_finished(self):
        self.modelsChanged.emit()
        self.close()

    @pyqtSlot()
    def onModelCreateClicked(self):
        self.createModelButton.setDisabled(True)

        name = self.outputModelName.text()
        vocab_size = self.vocabSize.value()
        hidden_dim = self.hiddenDim.value()
        latent_dim = self.latentDim.value()
        kl_step = self.klStep.value()
        dropout_prob = self.dropout.value()

        self.thread = QThread()
        self.worker = CreateModelWorker(vocab_size, hidden_dim, latent_dim, kl_step, dropout_prob, name)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.exit_on_worker_finished)

        self.thread.start()

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot

import controllers.controller
from gui.DatasetWidget import DatasetWidget
from gui.designer.Ui_ModelWizard import Ui_ModelWizard


class ModelWizard(QtWidgets.QDialog, Ui_ModelWizard):
    def __init__(self, datasets: QtWidgets.QListWidget, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        num_items = datasets.count()
        for i in range(num_items):
            item = datasets.item(i)
            widget = datasets.itemWidget(item).clone()

            new_item = QtWidgets.QListWidgetItem()
            new_item.setSizeHint(widget.sizeHint())
            self.datasetsChooseList.addItem(new_item)
            self.datasetsChooseList.setItemWidget(new_item, widget)

        self.infoStyle = 'QLabel { color: black; font: "Segoe UI"; font-size: 13px }'
        self.warningStyle = 'QLabel { color: red; font: "Segoe UI"; font-size: 13px }'

        # SIGNAL TO INFER PARAMS
        self.datasetsChooseList.itemDoubleClicked.connect(self.inferVocabSize)

        # CHANGED OUTPUT NAME
        self.outputModelName.textChanged.connect(self.checkModelOverwrite)

    @pyqtSlot(QtWidgets.QListWidgetItem)
    def inferVocabSize(self, item: QtWidgets.QListWidgetItem):
        widget: DatasetWidget = self.datasetsChooseList.itemWidget(item)
        vocab_size = int(widget.vocabulary.text().replace(",", ""))
        self.outputModelName.setText(widget.name.text())
        self.vocabSize.setValue(vocab_size)

    @pyqtSlot(str)
    def checkModelOverwrite(self, output_name: str):
        meta_info = controllers.controller.check_model_available(output_name)
        if meta_info:
            self.statusMessage.setStyleSheet(self.warningStyle)
            self.statusMessage.setText(f"Creating model will overwrite the previous one. Change name if you wish to avoid this!")
        else:
            self.statusMessage.setStyleSheet(self.infoStyle)
            self.statusMessage.setText("Fill in parameters...")

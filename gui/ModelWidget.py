import sys

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QListWidgetItem, QListWidget

from controllers.usersetup import load_config
from storage.MetaInfo import ModelMetaInfo


class ModelWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ModelWidget, self).__init__(parent)

        self.hLayout = QtWidgets.QHBoxLayout()

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(0, 0, 0, 0)

        self.hboxLayout = QtWidgets.QHBoxLayout()

        # ICON

        self.iconLabel = QLabel("model-icon")

        # LABELS

        self.label_font = QFont()
        self.label_font.setBold(False)
        self.label_font.setWeight(75)

        self.nameLabel = QLabel("Name")
        self.vocabLabel = QLabel("Vocabulary dim")
        self.hiddenDimLabel = QLabel("Hidden dim")
        self.latentDimLabel = QLabel("Latent dim")
        self.klStepLabel = QLabel("KL step")
        self.dropoutLabel = QLabel("Dropout prob")
        self.fitLabel = QLabel("Fit dataset")

        self.labels = [self.nameLabel, self.vocabLabel, self.hiddenDimLabel, self.latentDimLabel,
                       self.klStepLabel, self.dropoutLabel, self.fitLabel]

        for label in self.labels:
            label.setFont(self.label_font)

        # FIELDS

        self.name = QLabel("loading...")
        self.vocab = QLabel("loading...")
        self.hiddenDim = QLabel("loading...")
        self.latentDim = QLabel("loading...")
        self.klStep = QLabel("loading...")
        self.dropout = QLabel("loading...")
        self.fit = QLabel("loading...")

        # FIELDS AND STYLES

        self.fields = [self.name, self.vocab, self.hiddenDim, self.latentDim, self.klStep, self.dropout, self.fit]

        for f in self.fields:
            f.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
            f.setMinimumSize(100, 20)

        self.fieldStyle = 'QLabel { color: black; font: "Segoe UI"; font-size: 14px }'
        self.fieldErrorStyle = 'QLabel { color: red; font: bold "Segoe UI"; font-size: 14px }'

        # ADDING ROWS

        self.formLayout.addRow(self.nameLabel, self.name)
        self.formLayout.addRow(self.vocabLabel, self.vocab)
        self.formLayout.addRow(self.hiddenDimLabel, self.hiddenDim)
        self.formLayout.addRow(self.latentDimLabel, self.latentDim)
        self.formLayout.addRow(self.klStepLabel, self.klStep)
        self.formLayout.addRow(self.dropoutLabel, self.dropout)
        self.formLayout.addRow(self.fitLabel, self.fit)

        # WRAPPING HBOX

        self.hboxLayout.addWidget(self.iconLabel, 0)
        self.hboxLayout.addLayout(self.formLayout, 1)

        # MAIN LAYOUT
        self.setLayout(self.hboxLayout)

        # SETUP
        self.reset_state()

    def set_default_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/model.png"))

    def set_bad_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/model-inactive.png"))

    def set_inactive_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/model-inactive.png"))

    def reset_state(self):
        for f in self.fields:
            f.setStyleSheet(self.fieldStyle)
        self.set_default_icon()

    def mark_not_parsable(self):
        self.set_bad_icon()

        for f in self.fields:
            f.setText("???")
            f.setStyleSheet(self.fieldErrorStyle)

    def mark_mismatch_error(self, kind: str = "vocab_size"):
        self.set_inactive_icon()

        if kind == "vocab_size":
            self.vocab.setStyleSheet(self.fieldErrorStyle)

    def mark_field_unknown(self, field):
        field.setStyleSheet(self.fieldErrorStyle)
        field.setText("???")

    def set_fields(self, mi: ModelMetaInfo):
        try:
            self.name.setText(mi.info["name"])
        except KeyError:
            self.mark_field_unknown(self.name)

        try:
            self.vocab.setText("{:,}".format(mi.info["vocab_size"]))
        except KeyError:
            self.mark_field_unknown(self.vocab)

        try:
            self.hiddenDim.setText("{:,}".format(mi.info["hidden_dim"]))
        except KeyError:
            self.mark_field_unknown(self.hiddenDim)

        try:
            self.latentDim.setText("{:,}".format(mi.info["latent_dim"]))
        except KeyError:
            self.mark_field_unknown(self.latentDim)

        try:
            self.klStep.setText(str(mi.info["kl_step"]))
        except KeyError:
            self.mark_field_unknown(self.klStep)

        try:
            self.dropout.setText(str(mi.info["dropout_prob"]))
        except KeyError:
            self.mark_field_unknown(self.dropout)

        try:
            fit_dataset = mi.info["fit_dataset"]
            if fit_dataset:
                self.fit.setText(fit_dataset)
            else:
                self.fit.setText("---")
        except KeyError:
            self.mark_field_unknown(self.fit)


class exampleQMainWindow(QMainWindow):
    def __init__(self):
        super(exampleQMainWindow, self).__init__()
        # Create QListWidget

        self.datasetList = QListWidget(self)

        model1 = ModelWidget(self.datasetList)

        item = QListWidgetItem(self.datasetList)
        item.setSizeHint(model1.sizeHint())

        self.datasetList.addItem(item)
        self.datasetList.setItemWidget(item, model1)

        self.setCentralWidget(self.datasetList)

        mi = ModelMetaInfo.from_file(load_config()["model"]["model_home"] + "/20ng_user")
        model1.set_fields(mi)

        self.setCentralWidget(self.datasetList)

        self.setMinimumSize(300, 300)

        """
        d1 = DatasetWidget(self.datasetList)

        item = QListWidgetItem(self.datasetList)
        item.setSizeHint(d1.sizeHint())

        self.datasetList.addItem(item)
        self.datasetList.setItemWidget(item, d1)

        self.setCentralWidget(self.datasetList)

        mi = DatasetMetaInfo.from_file(load_config()["model"]["data_home"] + "/20ng_user")
        d1.set_fields(mi)
        """


if __name__ == "__main__":
    app = QApplication([])
    window = exampleQMainWindow()
    window.show()

    sys.exit(app.exec_())

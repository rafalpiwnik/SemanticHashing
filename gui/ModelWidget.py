from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QLabel, QMenu, QAction

from controllers import controller
from gui.EntityWidget import EntityWidget
from storage.MetaInfo import ModelMetaInfo


class ModelWidget(QtWidgets.QWidget, EntityWidget):
    modelRemoved = pyqtSignal()

    def __init__(self, parent=None):
        super(ModelWidget, self).__init__(parent)

        self.hLayout = QtWidgets.QHBoxLayout()

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(0, 0, 0, 0)

        self.hboxLayout = QtWidgets.QHBoxLayout()
        self.vboxCentering = QtWidgets.QVBoxLayout()

        # ICON
        self.iconLabel = QLabel("model-icon")

        # LABELS
        self.nameLabel = QLabel("Name")
        self.vocabLabel = QLabel("Vocabulary dim")
        self.hiddenDimLabel = QLabel("Hidden dim")
        self.latentDimLabel = QLabel("Latent dim")
        self.klStepLabel = QLabel("KL step")
        self.dropoutLabel = QLabel("Dropout prob")
        self.fitLabel = QLabel("Fit dataset")
        self.vectorizerLabel = QLabel("Vectorizer")

        self.labels = [self.nameLabel, self.vocabLabel, self.hiddenDimLabel, self.latentDimLabel,
                       self.klStepLabel, self.dropoutLabel, self.fitLabel, self.vectorizerLabel]

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
        self.vectorizer = QLabel("loading...")

        # FIELDS AND STYLES
        self.fields = [self.name, self.vocab, self.hiddenDim, self.latentDim, self.klStep, self.dropout, self.fit,
                       self.vectorizer]

        for f in self.fields:
            f.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # ADDING ROWS
        self.formLayout.addRow(self.nameLabel, self.name)
        self.formLayout.addRow(self.vocabLabel, self.vocab)
        self.formLayout.addRow(self.hiddenDimLabel, self.hiddenDim)
        self.formLayout.addRow(self.latentDimLabel, self.latentDim)
        self.formLayout.addRow(self.klStepLabel, self.klStep)
        self.formLayout.addRow(self.dropoutLabel, self.dropout)
        self.formLayout.addRow(self.fitLabel, self.fit)
        self.formLayout.addRow(self.vectorizerLabel, self.vectorizer)

        # WRAPPING HBOX
        self.hboxLayout.addWidget(self.iconLabel, 0)
        self.hboxLayout.addLayout(self.formLayout, 1)

        self.hboxLayout.setAlignment(Qt.AlignVCenter)

        # MAIN LAYOUT
        self.setLayout(self.hboxLayout)

        # SETUP
        self.reset_state()

    def clone(self):
        copy = ModelWidget()
        for src_field, dest_field in zip(self.fields, copy.fields):
            dest_field.setText(src_field.text())
        copy.setContextMenuPolicy(Qt.PreventContextMenu)
        return copy

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QMenu(self)

        remove = QAction("Remove", self)
        remove.setIcon(QIcon("../resources/icon-delete.png"))
        remove.triggered.connect(self.make_remove_model)
        menu.addAction(remove)

        menu.exec_(self.mapToGlobal(event.pos()))

    @pyqtSlot()
    def make_remove_model(self):
        success = controller.remove_model(self.name.text())
        if success:
            self.modelRemoved.emit()
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage("Cannot remove dataset")

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

    def mark_mismatch_error(self, kind: str = "vocab_size"):
        self.set_inactive_icon()

        if kind == "vocab_size":
            self.vocab.setStyleSheet(self.fieldErrorStyle)

    def mark_native(self):
        self.fit.setStyleSheet(self.fieldMatchStyle)

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

        try:
            has_vectorizer = controller.check_model_has_vectorizer(model_name=mi.name)
            if has_vectorizer:
                self.vectorizer.setText("Present")
            else:
                self.vectorizer.setText("Not defined")
        except KeyError:
            self.mark_field_unknown(self.dropout)

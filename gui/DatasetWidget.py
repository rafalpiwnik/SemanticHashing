import sys

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QListWidgetItem, QListWidget, QMenu, QAction, \
    QGraphicsOpacityEffect

from controllers import controller
from gui.EntityWidget import EntityWidget
from storage import entity_discovery
from storage.MetaInfo import DatasetMetaInfo


class DatasetWidget(QtWidgets.QWidget, EntityWidget):
    datasetRemoved = pyqtSignal()

    def __init__(self, parent=None):
        super(DatasetWidget, self).__init__(parent)

        self.hLayout = QtWidgets.QHBoxLayout()

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(0, 0, 0, 0)

        self.hboxLayout = QtWidgets.QHBoxLayout()

        # ICON

        self.iconLabel = QLabel("dataset-icon")

        # LABELS

        self.nameLabel = QLabel("Name")
        self.vocabLabel = QLabel("Vocabulary size")
        self.trainLabel = QLabel("Train size")
        self.testLabel = QLabel("Test size")
        self.kindLabel = QLabel("Label kind")
        self.authorLabel = QLabel("Author")

        self.labels = [self.nameLabel, self.vocabLabel, self.trainLabel, self.testLabel, self.kindLabel,
                       self.authorLabel]

        for lab in self.labels:
            lab.setFont(self.label_font)

        # FIELDS

        self.name = QLabel("loading...")
        self.train = QLabel("loading...")
        self.test = QLabel("loading...")
        self.vocabulary = QLabel("loading...")
        self.kind = QLabel("loading...")
        self.author = QLabel("loading...")
        self.date = QLabel("loading...")

        self.fields = [self.name, self.train, self.test, self.vocabulary, self.kind, self.author, self.date]

        # Set style for fields
        for f in self.fields:
            f.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # ADDING ROWS

        self.formLayout.addRow(self.nameLabel, self.name)
        self.formLayout.addRow(self.vocabLabel, self.vocabulary)
        self.formLayout.addRow(self.trainLabel, self.train)
        self.formLayout.addRow(self.testLabel, self.test)
        self.formLayout.addRow(self.kindLabel, self.kind)
        self.formLayout.addRow(self.authorLabel, self.author)

        # WRAPPING HBOX

        self.hboxLayout.addWidget(self.iconLabel, 0)
        self.hboxLayout.addLayout(self.formLayout, 1)

        # MAIN LAYOUT
        self.setLayout(self.hboxLayout)

        # SETUP
        self.reset_state()

    def clone(self):
        # ICON left Unchanged
        copy = DatasetWidget()
        for src_field, dest_field in zip(self.fields, copy.fields):
            dest_field.setText(src_field.text())
        return copy

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        # Can open context menu only when main window is set
        menu = QMenu(self)

        openSrc = QAction("Open location", self)
        remove = QAction("Remove", self)

        remove.triggered.connect(self.make_remove_dataset)

        menu.addAction(openSrc)
        menu.addAction(remove)

        action = menu.exec_(self.mapToGlobal(event.pos()))

    @pyqtSlot()
    def make_remove_dataset(self):
        success = controller.remove_dataset(self.name.text())
        if success:
            self.datasetRemoved.emit()
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage("Cannot remove dataset")

    def set_default_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/dataset.png"))

    def set_bad_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/dataset-inactive-bad.png"))

    def set_inactive_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/dataset-inactive.png"))

    def reset_state(self):
        for f in self.fields:
            f.setStyleSheet(self.fieldStyle)
        self.set_default_icon()

    def mark_mismatch_error(self, kind: str = "vocab_size"):
        self.set_inactive_icon()

        if kind == "vocab_size":
            self.vocabulary.setStyleSheet(self.fieldErrorStyle)

    def set_fields(self, mi: DatasetMetaInfo):
        try:
            self.name.setText(mi.info["name"])
        except KeyError:
            self.mark_field_unknown(self.name)

        try:
            if mi.info["user_author"]:
                self.author.setText("user")
            else:
                self.author.setText("fetched")
        except KeyError:
            self.mark_field_unknown(self.author)

        try:
            self.train.setText("{:,}".format(mi.info["num_train"]))
        except KeyError:
            self.mark_field_unknown(self.train)

        try:
            self.test.setText("{:,}".format(mi.info["num_test"]))
        except KeyError:
            self.mark_field_unknown(self.test)

        try:
            self.vocabulary.setText("{:,}".format(mi.info["vocab_size"]))
        except KeyError:
            self.mark_field_unknown(self.vocabulary)

        try:
            num_labels = mi.info["num_labels"]
            if num_labels == 0:
                self.kind.setText("unlabeled")
            elif num_labels == 1:
                self.kind.setText("single-labeled")
            elif num_labels > 1:
                self.kind.setText(f"multi-labeled ({num_labels})")
        except KeyError:
            self.mark_field_unknown(self.kind)


class exampleQMainWindow(QMainWindow):
    def __init__(self):
        super(exampleQMainWindow, self).__init__()
        # Create QListWidget

        self.datasetList = QListWidget(self)

        self.scan_fill()

        self.setCentralWidget(self.datasetList)

        self.setMinimumSize(400, 1000)

    def scan_fill(self):
        dss = entity_discovery.scan_datasets()

        self.datasetList.clear()

        for ds in dss:
            dataset_widget = DatasetWidget(self.datasetList)
            dataset_widget.set_fields(ds)

            item = QListWidgetItem(self.datasetList)
            item.setSizeHint(dataset_widget.sizeHint())

            self.datasetList.addItem(item)
            self.datasetList.setItemWidget(item, dataset_widget)


if __name__ == "__main__":
    app = QApplication([])
    window = exampleQMainWindow()
    window.show()

    sys.exit(app.exec_())

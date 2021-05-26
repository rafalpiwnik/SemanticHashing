import sys
from time import sleep

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QListWidgetItem, QListWidget

import controllers.entity_discovery
from preprocess.MetaInfo import DatasetMetaInfo


class DatasetWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DatasetWidget, self).__init__(parent)

        self.hLayout = QtWidgets.QHBoxLayout()

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(0, 0, 0, 0)

        self.hboxLayout = QtWidgets.QHBoxLayout()

        # ICON

        self.iconLabel = QLabel("dataset-icon")

        # LABELS

        self.label_font = QFont()
        self.label_font.setBold(False)
        self.label_font.setWeight(75)

        self.nameLabel = QLabel("Name")
        self.nameLabel.setFont(self.label_font)

        self.vocabLabel = QLabel("Vocabulary size")
        self.vocabLabel.setFont(self.label_font)

        self.trainLabel = QLabel("Train size")
        self.trainLabel.setFont(self.label_font)

        self.testLabel = QLabel("Test size")
        self.testLabel.setFont(self.label_font)

        self.kindLabel = QLabel("Label kind")
        self.kindLabel.setFont(self.label_font)

        self.authorLabel = QLabel("Author")
        self.authorLabel.setFont(self.label_font)

        # FIELDS

        self.name = QLabel("loading...")

        self.train = QLabel("loading...")

        self.test = QLabel("loading...")

        self.vocabulary = QLabel("loading...")

        self.kind = QLabel("loading...")

        self.author = QLabel("loading...")

        self.date = QLabel("loading...")

        # FIELDS AND STYLES

        self.fields = [self.name, self.train, self.test, self.vocabulary, self.kind, self.author, self.date]

        for f in self.fields:
            f.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
            f.setMinimumSize(100, 20)

        self.fieldStyle = 'QLabel { color: black; font: "Segoe UI"; font-size: 14px }'
        self.fieldErrorStyle = 'QLabel { color: red; font: bold "Segoe UI"; font-size: 14px }'

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

    def set_default_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/dataset.png"))

    def set_bad_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/dataset-bad.png"))

    def set_inactive_icon(self):
        self.iconLabel.setPixmap(QtGui.QPixmap("../resources/dataset-inactive.png"))

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
            self.vocabulary.setStyleSheet(self.fieldErrorStyle)

    def mark_field_unknown(self, field):
        field.setStyleSheet(self.fieldErrorStyle)
        field.setText("???")

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

    def scan_fill(self):
        dss = controllers.entity_discovery.scan_datasets()

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

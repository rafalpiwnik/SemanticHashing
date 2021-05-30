# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TestDialog(object):
    def setupUi(self, TestDialog):
        TestDialog.setObjectName("TestDialog")
        TestDialog.resize(441, 126)
        self.gridLayout = QtWidgets.QGridLayout(TestDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.progressBar = QtWidgets.QProgressBar(TestDialog)
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(TestDialog)
        self.groupBox.setObjectName("groupBox")
        self.formLayout = QtWidgets.QFormLayout(self.groupBox)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.precision = QtWidgets.QLabel(self.groupBox)
        self.precision.setObjectName("precision")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.precision)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 2)
        self.finishTestButton = QtWidgets.QPushButton(TestDialog)
        self.finishTestButton.setObjectName("finishTestButton")
        self.gridLayout.addWidget(self.finishTestButton, 1, 1, 1, 1)
        self.statusMessage = QtWidgets.QLabel(TestDialog)
        self.statusMessage.setObjectName("statusMessage")
        self.gridLayout.addWidget(self.statusMessage, 2, 0, 1, 1)

        self.retranslateUi(TestDialog)
        QtCore.QMetaObject.connectSlotsByName(TestDialog)

    def retranslateUi(self, TestDialog):
        _translate = QtCore.QCoreApplication.translate
        TestDialog.setWindowTitle(_translate("TestDialog", "Precision and recall test"))
        self.groupBox.setTitle(_translate("TestDialog", "Metrics"))
        self.label.setText(_translate("TestDialog", "Mean precision"))
        self.precision.setText(_translate("TestDialog", "---"))
        self.finishTestButton.setText(_translate("TestDialog", "Finish"))
        self.statusMessage.setText(_translate("TestDialog", "Running test..."))
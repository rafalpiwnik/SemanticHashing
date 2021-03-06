# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SearchDialog(object):
    def setupUi(self, SearchDialog):
        SearchDialog.setObjectName("SearchDialog")
        SearchDialog.resize(1296, 620)
        self.gridLayout = QtWidgets.QGridLayout(SearchDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.statusMessage = QtWidgets.QLabel(SearchDialog)
        self.statusMessage.setObjectName("statusMessage")
        self.gridLayout.addWidget(self.statusMessage, 2, 0, 1, 1)
        self.finishButton = QtWidgets.QPushButton(SearchDialog)
        self.finishButton.setEnabled(False)
        self.finishButton.setMaximumSize(QtCore.QSize(100, 16777215))
        self.finishButton.setFlat(False)
        self.finishButton.setObjectName("finishButton")
        self.gridLayout.addWidget(self.finishButton, 2, 1, 1, 1)
        self.fileTable = QtWidgets.QTableWidget(SearchDialog)
        self.fileTable.setEnabled(True)
        self.fileTable.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.fileTable.setMidLineWidth(0)
        self.fileTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.fileTable.setAlternatingRowColors(True)
        self.fileTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.fileTable.setTextElideMode(QtCore.Qt.ElideRight)
        self.fileTable.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.fileTable.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.fileTable.setShowGrid(True)
        self.fileTable.setGridStyle(QtCore.Qt.SolidLine)
        self.fileTable.setObjectName("fileTable")
        self.fileTable.setColumnCount(4)
        self.fileTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.fileTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.fileTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.fileTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.fileTable.setHorizontalHeaderItem(3, item)
        self.fileTable.horizontalHeader().setCascadingSectionResizes(True)
        self.fileTable.horizontalHeader().setHighlightSections(False)
        self.fileTable.horizontalHeader().setMinimumSectionSize(150)
        self.fileTable.verticalHeader().setCascadingSectionResizes(False)
        self.fileTable.verticalHeader().setDefaultSectionSize(28)
        self.gridLayout.addWidget(self.fileTable, 0, 0, 1, 2)
        self.progbar = QtWidgets.QProgressBar(SearchDialog)
        self.progbar.setProperty("value", 0)
        self.progbar.setAlignment(QtCore.Qt.AlignCenter)
        self.progbar.setObjectName("progbar")
        self.gridLayout.addWidget(self.progbar, 1, 0, 1, 2)

        self.retranslateUi(SearchDialog)
        QtCore.QMetaObject.connectSlotsByName(SearchDialog)

    def retranslateUi(self, SearchDialog):
        _translate = QtCore.QCoreApplication.translate
        SearchDialog.setWindowTitle(_translate("SearchDialog", "VDSH file search"))
        self.statusMessage.setText(_translate("SearchDialog", "Starting search..."))
        self.finishButton.setText(_translate("SearchDialog", "Finish"))
        item = self.fileTable.horizontalHeaderItem(0)
        item.setText(_translate("SearchDialog", "Name"))
        item = self.fileTable.horizontalHeaderItem(1)
        item.setText(_translate("SearchDialog", "Path"))
        item = self.fileTable.horizontalHeaderItem(2)
        item.setText(_translate("SearchDialog", "Hamming distance"))
        item = self.fileTable.horizontalHeaderItem(3)
        item.setText(_translate("SearchDialog", "Binary code"))

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'semhash_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1102, 1026)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 300))
        self.tabWidget.setBaseSize(QtCore.QSize(0, 350))
        self.tabWidget.setObjectName("tabWidget")
        self.tabTrain = QtWidgets.QWidget()
        self.tabTrain.setObjectName("tabTrain")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tabTrain)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.trainOptionsGridLayout = QtWidgets.QGridLayout()
        self.trainOptionsGridLayout.setObjectName("trainOptionsGridLayout")
        self.groupBoxTrainParams = QtWidgets.QGroupBox(self.tabTrain)
        self.groupBoxTrainParams.setObjectName("groupBoxTrainParams")
        self.formLayout = QtWidgets.QFormLayout(self.groupBoxTrainParams)
        self.formLayout.setObjectName("formLayout")
        self.trainEpochsLabel = QtWidgets.QLabel(self.groupBoxTrainParams)
        self.trainEpochsLabel.setObjectName("trainEpochsLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.trainEpochsLabel)
        self.trainEpochs = QtWidgets.QSpinBox(self.groupBoxTrainParams)
        self.trainEpochs.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.trainEpochs.setProperty("showGroupSeparator", False)
        self.trainEpochs.setMinimum(1)
        self.trainEpochs.setMaximum(99)
        self.trainEpochs.setProperty("value", 25)
        self.trainEpochs.setDisplayIntegerBase(10)
        self.trainEpochs.setObjectName("trainEpochs")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.trainEpochs)
        self.trainBatchLabel = QtWidgets.QLabel(self.groupBoxTrainParams)
        self.trainBatchLabel.setObjectName("trainBatchLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.trainBatchLabel)
        self.trainBatch = QtWidgets.QSpinBox(self.groupBoxTrainParams)
        self.trainBatch.setObjectName("trainBatch")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.trainBatch)
        self.trainOptionsGridLayout.addWidget(self.groupBoxTrainParams, 1, 0, 1, 1)
        self.groupBoxTrainOptimizer = QtWidgets.QGroupBox(self.tabTrain)
        self.groupBoxTrainOptimizer.setObjectName("groupBoxTrainOptimizer")
        self.formLayout_2 = QtWidgets.QFormLayout(self.groupBoxTrainOptimizer)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label = QtWidgets.QLabel(self.groupBoxTrainOptimizer)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox = QtWidgets.QComboBox(self.groupBoxTrainOptimizer)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox)
        self.trainOptionsGridLayout.addWidget(self.groupBoxTrainOptimizer, 2, 0, 1, 1)
        self.gridLayout_3.addLayout(self.trainOptionsGridLayout, 2, 0, 1, 4)
        self.trainMixingGridLayout = QtWidgets.QGridLayout()
        self.trainMixingGridLayout.setHorizontalSpacing(50)
        self.trainMixingGridLayout.setObjectName("trainMixingGridLayout")
        self.trainModelGroupbox = QtWidgets.QGroupBox(self.tabTrain)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trainModelGroupbox.sizePolicy().hasHeightForWidth())
        self.trainModelGroupbox.setSizePolicy(sizePolicy)
        self.trainModelGroupbox.setObjectName("trainModelGroupbox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.trainModelGroupbox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.trainModelSpace = QtWidgets.QVBoxLayout()
        self.trainModelSpace.setObjectName("trainModelSpace")
        self.gridLayout_5.addLayout(self.trainModelSpace, 0, 0, 1, 1)
        self.trainMixingGridLayout.addWidget(self.trainModelGroupbox, 0, 2, 1, 1)
        self.trainDatasetGroupbox = QtWidgets.QGroupBox(self.tabTrain)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trainDatasetGroupbox.sizePolicy().hasHeightForWidth())
        self.trainDatasetGroupbox.setSizePolicy(sizePolicy)
        self.trainDatasetGroupbox.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.trainDatasetGroupbox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.trainDatasetGroupbox.setFlat(False)
        self.trainDatasetGroupbox.setCheckable(False)
        self.trainDatasetGroupbox.setObjectName("trainDatasetGroupbox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.trainDatasetGroupbox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.trainDatasetSpace = QtWidgets.QVBoxLayout()
        self.trainDatasetSpace.setObjectName("trainDatasetSpace")
        self.gridLayout_2.addLayout(self.trainDatasetSpace, 0, 1, 1, 1)
        self.trainMixingGridLayout.addWidget(self.trainDatasetGroupbox, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(100, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trainMixingGridLayout.addItem(spacerItem, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(100, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trainMixingGridLayout.addItem(spacerItem1, 0, 3, 1, 1)
        self.gridLayout_3.addLayout(self.trainMixingGridLayout, 1, 0, 1, 4)
        self.tabWidget.addTab(self.tabTrain, "")
        self.tabTest = QtWidgets.QWidget()
        self.tabTest.setObjectName("tabTest")
        self.tabWidget.addTab(self.tabTest, "")
        self.tabSearch = QtWidgets.QWidget()
        self.tabSearch.setObjectName("tabSearch")
        self.tabWidget.addTab(self.tabSearch, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.modelList = QtWidgets.QListWidget(self.centralwidget)
        self.modelList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.modelList.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.modelList.setAlternatingRowColors(False)
        self.modelList.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.modelList.setObjectName("modelList")
        self.gridLayout.addWidget(self.modelList, 1, 1, 1, 1)
        self.datasetList = QtWidgets.QListWidget(self.centralwidget)
        self.datasetList.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.datasetList.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.datasetList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.datasetList.setAlternatingRowColors(False)
        self.datasetList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.datasetList.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.datasetList.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.datasetList.setUniformItemSizes(True)
        self.datasetList.setSelectionRectVisible(False)
        self.datasetList.setObjectName("datasetList")
        self.gridLayout.addWidget(self.datasetList, 1, 0, 1, 1)
        self.modelsLabel = QtWidgets.QLabel(self.centralwidget)
        self.modelsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.modelsLabel.setObjectName("modelsLabel")
        self.gridLayout.addWidget(self.modelsLabel, 0, 1, 1, 1)
        self.datasetsLabel = QtWidgets.QLabel(self.centralwidget)
        self.datasetsLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.datasetsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.datasetsLabel.setObjectName("datasetsLabel")
        self.gridLayout.addWidget(self.datasetsLabel, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1102, 26))
        self.menubar.setObjectName("menubar")
        self.menuFiles = QtWidgets.QMenu(self.menubar)
        self.menuFiles.setObjectName("menuFiles")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionSettings = QtWidgets.QAction(MainWindow)
        self.actionSettings.setObjectName("actionSettings")
        self.actionNew_dataset = QtWidgets.QAction(MainWindow)
        self.actionNew_dataset.setObjectName("actionNew_dataset")
        self.actionNew_model = QtWidgets.QAction(MainWindow)
        self.actionNew_model.setObjectName("actionNew_model")
        self.menuFiles.addAction(self.actionSettings)
        self.menubar.addAction(self.menuFiles.menuAction())
        self.toolBar.addAction(self.actionNew_dataset)
        self.toolBar.addAction(self.actionNew_model)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBoxTrainParams.setTitle(_translate("MainWindow", "Parameters"))
        self.trainEpochsLabel.setText(_translate("MainWindow", "Epochs"))
        self.trainBatchLabel.setText(_translate("MainWindow", "Batch size"))
        self.groupBoxTrainOptimizer.setTitle(_translate("MainWindow", "Optimizer"))
        self.label.setText(_translate("MainWindow", "Type"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Adam optimizer"))
        self.trainModelGroupbox.setTitle(_translate("MainWindow", "Model"))
        self.trainDatasetGroupbox.setTitle(_translate("MainWindow", "Dataset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTrain), _translate("MainWindow", "Train model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTest), _translate("MainWindow", "Test model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabSearch), _translate("MainWindow", "File search"))
        self.modelsLabel.setText(_translate("MainWindow", "Models"))
        self.datasetsLabel.setText(_translate("MainWindow", "Datasets"))
        self.menuFiles.setTitle(_translate("MainWindow", "Files"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))
        self.actionNew_dataset.setText(_translate("MainWindow", "New dataset"))
        self.actionNew_dataset.setToolTip(_translate("MainWindow", "<html><head/><body><p>Create a new dataset</p></body></html>"))
        self.actionNew_model.setText(_translate("MainWindow", "New model"))
        self.actionNew_model.setToolTip(_translate("MainWindow", "<html><head/><body><p>Create a new VDSH model</p></body></html>"))
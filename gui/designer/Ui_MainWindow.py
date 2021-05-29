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
        MainWindow.resize(1064, 1140)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(9, 3, 9, 3)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 300))
        self.tabWidget.setMaximumSize(QtCore.QSize(16777215, 300))
        self.tabWidget.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tabTrain = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabTrain.sizePolicy().hasHeightForWidth())
        self.tabTrain.setSizePolicy(sizePolicy)
        self.tabTrain.setObjectName("tabTrain")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tabTrain)
        self.gridLayout_3.setObjectName("gridLayout_3")
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
        self.trainModelStack = QtWidgets.QStackedWidget(self.trainModelGroupbox)
        self.trainModelStack.setObjectName("trainModelStack")
        self.trainModelSpace.addWidget(self.trainModelStack)
        self.gridLayout_5.addLayout(self.trainModelSpace, 0, 0, 1, 1)
        self.trainMixingGridLayout.addWidget(self.trainModelGroupbox, 1, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(100, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trainMixingGridLayout.addItem(spacerItem, 1, 0, 1, 1)
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
        self.trainDatasetStack = QtWidgets.QStackedWidget(self.trainDatasetGroupbox)
        self.trainDatasetStack.setObjectName("trainDatasetStack")
        self.trainDatasetSpace.addWidget(self.trainDatasetStack)
        self.gridLayout_2.addLayout(self.trainDatasetSpace, 0, 1, 1, 1)
        self.trainMixingGridLayout.addWidget(self.trainDatasetGroupbox, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(100, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trainMixingGridLayout.addItem(spacerItem1, 1, 3, 1, 1)
        self.gridLayout_3.addLayout(self.trainMixingGridLayout, 1, 0, 1, 2)
        self.trainTabInterface = QtWidgets.QGridLayout()
        self.trainTabInterface.setObjectName("trainTabInterface")
        self.buttonTrainWizard = QtWidgets.QPushButton(self.tabTrain)
        self.buttonTrainWizard.setMaximumSize(QtCore.QSize(100, 16777215))
        self.buttonTrainWizard.setCheckable(False)
        self.buttonTrainWizard.setAutoDefault(False)
        self.buttonTrainWizard.setDefault(False)
        self.buttonTrainWizard.setFlat(False)
        self.buttonTrainWizard.setObjectName("buttonTrainWizard")
        self.trainTabInterface.addWidget(self.buttonTrainWizard, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.trainTabInterface, 3, 0, 1, 2)
        self.tabWidget.addTab(self.tabTrain, "")
        self.tabTest = QtWidgets.QWidget()
        self.tabTest.setObjectName("tabTest")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tabTest)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.testMixingGridLayout = QtWidgets.QGridLayout()
        self.testMixingGridLayout.setHorizontalSpacing(50)
        self.testMixingGridLayout.setObjectName("testMixingGridLayout")
        self.testModelGroupbox = QtWidgets.QGroupBox(self.tabTest)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.testModelGroupbox.sizePolicy().hasHeightForWidth())
        self.testModelGroupbox.setSizePolicy(sizePolicy)
        self.testModelGroupbox.setObjectName("testModelGroupbox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.testModelGroupbox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.testModelSpace = QtWidgets.QVBoxLayout()
        self.testModelSpace.setObjectName("testModelSpace")
        self.testModelStack = QtWidgets.QStackedWidget(self.testModelGroupbox)
        self.testModelStack.setObjectName("testModelStack")
        self.testModelSpace.addWidget(self.testModelStack)
        self.gridLayout_6.addLayout(self.testModelSpace, 0, 0, 1, 1)
        self.testMixingGridLayout.addWidget(self.testModelGroupbox, 1, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(100, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.testMixingGridLayout.addItem(spacerItem2, 1, 0, 1, 1)
        self.testDatasetGroupbox = QtWidgets.QGroupBox(self.tabTest)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.testDatasetGroupbox.sizePolicy().hasHeightForWidth())
        self.testDatasetGroupbox.setSizePolicy(sizePolicy)
        self.testDatasetGroupbox.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.testDatasetGroupbox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.testDatasetGroupbox.setFlat(False)
        self.testDatasetGroupbox.setCheckable(False)
        self.testDatasetGroupbox.setObjectName("testDatasetGroupbox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.testDatasetGroupbox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.testDatasetSpace = QtWidgets.QVBoxLayout()
        self.testDatasetSpace.setObjectName("testDatasetSpace")
        self.testDatasetStack = QtWidgets.QStackedWidget(self.testDatasetGroupbox)
        self.testDatasetStack.setObjectName("testDatasetStack")
        self.testDatasetSpace.addWidget(self.testDatasetStack)
        self.gridLayout_4.addLayout(self.testDatasetSpace, 0, 1, 1, 1)
        self.testMixingGridLayout.addWidget(self.testDatasetGroupbox, 1, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(100, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.testMixingGridLayout.addItem(spacerItem3, 1, 3, 1, 1)
        self.verticalLayout_2.addLayout(self.testMixingGridLayout)
        self.tabWidget.addTab(self.tabTest, "")
        self.tabSearch = QtWidgets.QWidget()
        self.tabSearch.setObjectName("tabSearch")
        self.tabWidget.addTab(self.tabSearch, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setVerticalSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.datasetList = QtWidgets.QListWidget(self.centralwidget)
        self.datasetList.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.datasetList.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.datasetList.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.datasetList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.datasetList.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.datasetList.setAlternatingRowColors(False)
        self.datasetList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.datasetList.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.datasetList.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.datasetList.setUniformItemSizes(True)
        self.datasetList.setSelectionRectVisible(False)
        self.datasetList.setObjectName("datasetList")
        self.gridLayout.addWidget(self.datasetList, 1, 2, 1, 1)
        self.modelList = QtWidgets.QListWidget(self.centralwidget)
        self.modelList.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.modelList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.modelList.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.modelList.setAlternatingRowColors(False)
        self.modelList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.modelList.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.modelList.setObjectName("modelList")
        self.gridLayout.addWidget(self.modelList, 1, 3, 1, 1)
        self.datasetsLabel = QtWidgets.QLabel(self.centralwidget)
        self.datasetsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.datasetsLabel.setObjectName("datasetsLabel")
        self.gridLayout.addWidget(self.datasetsLabel, 0, 2, 1, 1)
        self.modelsLabel = QtWidgets.QLabel(self.centralwidget)
        self.modelsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.modelsLabel.setObjectName("modelsLabel")
        self.gridLayout.addWidget(self.modelsLabel, 0, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1064, 26))
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
        self.actionFetch_dataset = QtWidgets.QAction(MainWindow)
        self.actionFetch_dataset.setObjectName("actionFetch_dataset")
        self.actionOpen_home = QtWidgets.QAction(MainWindow)
        self.actionOpen_home.setObjectName("actionOpen_home")
        self.menuFiles.addAction(self.actionSettings)
        self.menuFiles.addAction(self.actionOpen_home)
        self.menubar.addAction(self.menuFiles.menuAction())
        self.toolBar.addAction(self.actionNew_dataset)
        self.toolBar.addAction(self.actionFetch_dataset)
        self.toolBar.addAction(self.actionNew_model)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.trainDatasetStack.setCurrentIndex(-1)
        self.testDatasetStack.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Semantic Hashing"))
        self.trainModelGroupbox.setTitle(_translate("MainWindow", "Model"))
        self.trainDatasetGroupbox.setTitle(_translate("MainWindow", "Dataset"))
        self.buttonTrainWizard.setText(_translate("MainWindow", "Train wizard"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTrain), _translate("MainWindow", "Train model"))
        self.testModelGroupbox.setTitle(_translate("MainWindow", "Model"))
        self.testDatasetGroupbox.setTitle(_translate("MainWindow", "Dataset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTest), _translate("MainWindow", "Test model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabSearch), _translate("MainWindow", "File search"))
        self.datasetsLabel.setText(_translate("MainWindow", "Datasets"))
        self.modelsLabel.setText(_translate("MainWindow", "Models"))
        self.menuFiles.setTitle(_translate("MainWindow", "Files"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))
        self.actionNew_dataset.setText(_translate("MainWindow", "New dataset"))
        self.actionNew_dataset.setToolTip(_translate("MainWindow", "<html><head/><body><p>Create a new dataset</p></body></html>"))
        self.actionNew_model.setText(_translate("MainWindow", "New model"))
        self.actionNew_model.setToolTip(_translate("MainWindow", "<html><head/><body><p>Create a new VDSH model</p></body></html>"))
        self.actionFetch_dataset.setText(_translate("MainWindow", "Fetch dataset"))
        self.actionFetch_dataset.setToolTip(_translate("MainWindow", "Fetch a vectorized dataset from an internet resource"))
        self.actionOpen_home.setText(_translate("MainWindow", "Open home"))

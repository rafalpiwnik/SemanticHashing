# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'model_wizard.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ModelWizard(object):
    def setupUi(self, ModelWizard):
        ModelWizard.setObjectName("ModelWizard")
        ModelWizard.resize(604, 547)
        self.gridLayout = QtWidgets.QGridLayout(ModelWizard)
        self.gridLayout.setObjectName("gridLayout")
        self.line_2 = QtWidgets.QFrame(ModelWizard)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.line_2.setLineWidth(3)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 1, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(ModelWizard)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.outputModelName = QtWidgets.QLineEdit(ModelWizard)
        self.outputModelName.setObjectName("outputModelName")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.outputModelName)
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)
        self.trainOptionsGridLayout = QtWidgets.QGridLayout()
        self.trainOptionsGridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.trainOptionsGridLayout.setObjectName("trainOptionsGridLayout")
        self.groupBoxTrainParams = QtWidgets.QGroupBox(ModelWizard)
        self.groupBoxTrainParams.setMaximumSize(QtCore.QSize(16777215, 16666666))
        self.groupBoxTrainParams.setObjectName("groupBoxTrainParams")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBoxTrainParams)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.formParameters = QtWidgets.QFormLayout()
        self.formParameters.setObjectName("formParameters")
        self.label_6 = QtWidgets.QLabel(self.groupBoxTrainParams)
        self.label_6.setObjectName("label_6")
        self.formParameters.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.vocabSize = QtWidgets.QSpinBox(self.groupBoxTrainParams)
        self.vocabSize.setMaximumSize(QtCore.QSize(1666666, 16777215))
        self.vocabSize.setMaximum(9999999)
        self.vocabSize.setSingleStep(2500)
        self.vocabSize.setProperty("value", 10000)
        self.vocabSize.setObjectName("vocabSize")
        self.formParameters.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.vocabSize)
        self.label_7 = QtWidgets.QLabel(self.groupBoxTrainParams)
        self.label_7.setObjectName("label_7")
        self.formParameters.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.hiddenDim = QtWidgets.QSpinBox(self.groupBoxTrainParams)
        self.hiddenDim.setMaximumSize(QtCore.QSize(1666666, 16777215))
        self.hiddenDim.setMaximum(999999)
        self.hiddenDim.setSingleStep(250)
        self.hiddenDim.setProperty("value", 1000)
        self.hiddenDim.setObjectName("hiddenDim")
        self.formParameters.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.hiddenDim)
        self.label_8 = QtWidgets.QLabel(self.groupBoxTrainParams)
        self.label_8.setObjectName("label_8")
        self.formParameters.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.latentDim = QtWidgets.QSpinBox(self.groupBoxTrainParams)
        self.latentDim.setMaximumSize(QtCore.QSize(1666666, 16777215))
        self.latentDim.setMinimum(1)
        self.latentDim.setMaximum(256)
        self.latentDim.setProperty("value", 32)
        self.latentDim.setObjectName("latentDim")
        self.formParameters.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.latentDim)
        self.label_9 = QtWidgets.QLabel(self.groupBoxTrainParams)
        self.label_9.setObjectName("label_9")
        self.formParameters.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.klStep = QtWidgets.QDoubleSpinBox(self.groupBoxTrainParams)
        self.klStep.setMaximumSize(QtCore.QSize(16666666, 16777215))
        self.klStep.setDecimals(5)
        self.klStep.setMinimum(0.0)
        self.klStep.setMaximum(1.0)
        self.klStep.setSingleStep(1e-05)
        self.klStep.setProperty("value", 0.0002)
        self.klStep.setObjectName("klStep")
        self.formParameters.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.klStep)
        self.label_11 = QtWidgets.QLabel(self.groupBoxTrainParams)
        self.label_11.setObjectName("label_11")
        self.formParameters.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.dropout = QtWidgets.QDoubleSpinBox(self.groupBoxTrainParams)
        self.dropout.setMaximumSize(QtCore.QSize(1666666, 1666666))
        self.dropout.setMaximum(1.0)
        self.dropout.setSingleStep(0.01)
        self.dropout.setProperty("value", 0.1)
        self.dropout.setObjectName("dropout")
        self.formParameters.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.dropout)
        self.gridLayout_2.addLayout(self.formParameters, 0, 0, 1, 1)
        self.inferVerticalLayout = QtWidgets.QVBoxLayout()
        self.inferVerticalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.inferVerticalLayout.setObjectName("inferVerticalLayout")
        self.label_10 = QtWidgets.QLabel(self.groupBoxTrainParams)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 16666))
        self.label_10.setObjectName("label_10")
        self.inferVerticalLayout.addWidget(self.label_10)
        self.datasetsChooseList = QtWidgets.QListWidget(self.groupBoxTrainParams)
        self.datasetsChooseList.setMaximumSize(QtCore.QSize(600, 220))
        self.datasetsChooseList.setFlow(QtWidgets.QListView.LeftToRight)
        self.datasetsChooseList.setObjectName("datasetsChooseList")
        self.inferVerticalLayout.addWidget(self.datasetsChooseList)
        self.gridLayout_2.addLayout(self.inferVerticalLayout, 2, 0, 1, 1)
        self.line = QtWidgets.QFrame(self.groupBoxTrainParams)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 1, 0, 1, 1)
        self.trainOptionsGridLayout.addWidget(self.groupBoxTrainParams, 1, 0, 1, 1)
        self.gridLayout.addLayout(self.trainOptionsGridLayout, 2, 0, 1, 1)
        self.createModelButton = QtWidgets.QPushButton(ModelWizard)
        self.createModelButton.setObjectName("createModelButton")
        self.gridLayout.addWidget(self.createModelButton, 5, 0, 1, 1)
        self.statusMessage = QtWidgets.QLabel(ModelWizard)
        self.statusMessage.setObjectName("statusMessage")
        self.gridLayout.addWidget(self.statusMessage, 6, 0, 1, 1)

        self.retranslateUi(ModelWizard)
        QtCore.QMetaObject.connectSlotsByName(ModelWizard)

    def retranslateUi(self, ModelWizard):
        _translate = QtCore.QCoreApplication.translate
        ModelWizard.setWindowTitle(_translate("ModelWizard", "Dialog"))
        self.label.setText(_translate("ModelWizard", "Output model name"))
        self.groupBoxTrainParams.setTitle(_translate("ModelWizard", "Parameters"))
        self.label_6.setText(_translate("ModelWizard", "Vocabulary size"))
        self.label_7.setText(_translate("ModelWizard", "Hidden dimension"))
        self.label_8.setText(_translate("ModelWizard", "Latent dimension"))
        self.label_9.setText(_translate("ModelWizard", "KL divergence weight step"))
        self.label_11.setText(_translate("ModelWizard", "Dropout probility"))
        self.label_10.setText(_translate("ModelWizard", "Infer vocabulary size from dataset"))
        self.createModelButton.setText(_translate("ModelWizard", "Create model"))
        self.statusMessage.setText(_translate("ModelWizard", "Fill in parameters..."))
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/getModels/SelectAI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1155, 645)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.InstallModelsFrame = QtWidgets.QFrame(self.centralwidget)
        self.InstallModelsFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.InstallModelsFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.InstallModelsFrame.setObjectName("InstallModelsFrame")
        self.gridLayout = QtWidgets.QGridLayout(self.InstallModelsFrame)
        self.gridLayout.setObjectName("gridLayout")
        self.label_13 = QtWidgets.QLabel(self.InstallModelsFrame)
        self.label_13.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 0, 0, 1, 1)
        self.InstallButton = QtWidgets.QPushButton(self.InstallModelsFrame)
        self.InstallButton.setObjectName("InstallButton")
        self.gridLayout.addWidget(self.InstallButton, 4, 0, 1, 1)
        self.installModelsProgressBar = QtWidgets.QProgressBar(self.InstallModelsFrame)
        self.installModelsProgressBar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.installModelsProgressBar.setProperty("value", 0)
        self.installModelsProgressBar.setTextVisible(False)
        self.installModelsProgressBar.setInvertedAppearance(False)
        self.installModelsProgressBar.setTextDirection(
            QtWidgets.QProgressBar.TopToBottom
        )
        self.installModelsProgressBar.setObjectName("installModelsProgressBar")
        self.gridLayout.addWidget(self.installModelsProgressBar, 5, 0, 1, 1)
        self.modelsTabWidget = QtWidgets.QTabWidget(self.InstallModelsFrame)
        self.modelsTabWidget.setObjectName("modelsTabWidget")
        self.NCNNTab = QtWidgets.QWidget()
        self.NCNNTab.setObjectName("NCNNTab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.NCNNTab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.RifeCheckBox = QtWidgets.QCheckBox(self.NCNNTab)
        self.RifeCheckBox.setObjectName("RifeCheckBox")
        self.horizontalLayout_2.addWidget(self.RifeCheckBox)
        self.RifeSettings = QtWidgets.QPushButton(self.NCNNTab)
        self.RifeSettings.setObjectName("RifeSettings")
        self.horizontalLayout_2.addWidget(self.RifeSettings)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.RealESRGANCheckBox = QtWidgets.QCheckBox(self.NCNNTab)
        self.RealESRGANCheckBox.setObjectName("RealESRGANCheckBox")
        self.verticalLayout_2.addWidget(self.RealESRGANCheckBox)
        self.RealSRCheckBox = QtWidgets.QCheckBox(self.NCNNTab)
        self.RealSRCheckBox.setEnabled(True)
        self.RealSRCheckBox.setObjectName("RealSRCheckBox")
        self.verticalLayout_2.addWidget(self.RealSRCheckBox)
        self.Waifu2xCheckBox = QtWidgets.QCheckBox(self.NCNNTab)
        self.Waifu2xCheckBox.setObjectName("Waifu2xCheckBox")
        self.verticalLayout_2.addWidget(self.Waifu2xCheckBox)
        self.RealCUGANCheckBox = QtWidgets.QCheckBox(self.NCNNTab)
        self.RealCUGANCheckBox.setEnabled(True)
        self.RealCUGANCheckBox.setObjectName("RealCUGANCheckBox")
        self.verticalLayout_2.addWidget(self.RealCUGANCheckBox)
        self.CainCheckBox = QtWidgets.QCheckBox(self.NCNNTab)
        self.CainCheckBox.setEnabled(True)
        self.CainCheckBox.setObjectName("CainCheckBox")
        self.verticalLayout_2.addWidget(self.CainCheckBox)
        self.RifeVSCheckBox = QtWidgets.QCheckBox(self.NCNNTab)
        self.RifeVSCheckBox.setEnabled(False)
        self.RifeVSCheckBox.setObjectName("RifeVSCheckBox")
        self.verticalLayout_2.addWidget(self.RifeVSCheckBox)
        self.modelsTabWidget.addTab(self.NCNNTab, "")
        self.CUDATab = QtWidgets.QWidget()
        self.CUDATab.setObjectName("CUDATab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.CUDATab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.RifeCUDACheckBox = QtWidgets.QCheckBox(self.CUDATab)
        self.RifeCUDACheckBox.setObjectName("RifeCUDACheckBox")
        self.verticalLayout.addWidget(self.RifeCUDACheckBox)
        self.RealESRGANCUDACheckBox = QtWidgets.QCheckBox(self.CUDATab)
        self.RealESRGANCUDACheckBox.setObjectName("RealESRGANCUDACheckBox")
        self.verticalLayout.addWidget(self.RealESRGANCUDACheckBox)
        self.modelsTabWidget.addTab(self.CUDATab, "")
        self.gridLayout.addWidget(self.modelsTabWidget, 1, 0, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout.addLayout(self.gridLayout_6, 2, 0, 1, 1)
        self.gridLayout_3.addWidget(self.InstallModelsFrame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.modelsTabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "REAL Video Enhancer"))
        self.label_13.setText(_translate("MainWindow", "Models:"))
        self.InstallButton.setText(_translate("MainWindow", "Install"))
        self.RifeCheckBox.setText(
            _translate("MainWindow", "RIFE (Interpolate General)")
        )
        self.RifeSettings.setText(_translate("MainWindow", "Settings"))
        self.RealESRGANCheckBox.setText(
            _translate("MainWindow", "RealESRGAN (Upscale Animation)")
        )
        self.RealSRCheckBox.setText(
            _translate("MainWindow", "RealSR (Upscale General)")
        )
        self.Waifu2xCheckBox.setText(
            _translate("MainWindow", "Waifu2X  (Upscale Animation)")
        )
        self.RealCUGANCheckBox.setText(
            _translate("MainWindow", "RealCUGAN (Upscale Animation)")
        )
        self.CainCheckBox.setText(
            _translate("MainWindow", "IFRNET (Interpolate General)")
        )
        self.RifeVSCheckBox.setText(
            _translate("MainWindow", "VS-Rife (Interpolate General)")
        )
        self.modelsTabWidget.setTabText(
            self.modelsTabWidget.indexOf(self.NCNNTab),
            _translate("MainWindow", "NCNN (AMD/Intel/Nvidia)"),
        )
        self.RifeCUDACheckBox.setText(
            _translate("MainWindow", "RIFE (Interpolate General)")
        )
        self.RealESRGANCUDACheckBox.setText(
            _translate("MainWindow", "RealESRGAN (Upscale Animation)")
        )
        self.modelsTabWidget.setTabText(
            self.modelsTabWidget.indexOf(self.CUDATab),
            _translate("MainWindow", "CUDA (Nvidia Only)"),
        )

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/getModels/SelectModels.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(973, 654)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        self.rife = QtWidgets.QCheckBox(self.centralwidget)
        self.rife.setObjectName("rife")
        self.verticalLayout_6.addWidget(self.rife)
        self.rifeanime = QtWidgets.QCheckBox(self.centralwidget)
        self.rifeanime.setObjectName("rifeanime")
        self.verticalLayout_6.addWidget(self.rifeanime)
        self.rifehd = QtWidgets.QCheckBox(self.centralwidget)
        self.rifehd.setObjectName("rifehd")
        self.verticalLayout_6.addWidget(self.rifehd)
        self.rifeuhd = QtWidgets.QCheckBox(self.centralwidget)
        self.rifeuhd.setObjectName("rifeuhd")
        self.verticalLayout_6.addWidget(self.rifeuhd)
        self.rife2 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife2.setObjectName("rife2")
        self.verticalLayout_6.addWidget(self.rife2)
        self.rife23 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife23.setObjectName("rife23")
        self.verticalLayout_6.addWidget(self.rife23)
        self.rife24 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife24.setObjectName("rife24")
        self.verticalLayout_6.addWidget(self.rife24)
        self.rife30 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife30.setObjectName("rife30")
        self.verticalLayout_6.addWidget(self.rife30)
        self.rife31 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife31.setObjectName("rife31")
        self.verticalLayout_6.addWidget(self.rife31)
        self.rife4 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife4.setObjectName("rife4")
        self.verticalLayout_6.addWidget(self.rife4)
        self.rife46 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife46.setObjectName("rife46")
        self.verticalLayout_6.addWidget(self.rife46)
        self.rife47 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife47.setObjectName("rife47")
        self.verticalLayout_6.addWidget(self.rife47)
        self.rife48 = QtWidgets.QCheckBox(self.centralwidget)
        self.rife48.setObjectName("rife48")
        self.verticalLayout_6.addWidget(self.rife48)
        self.horizontalLayout_2.addLayout(self.verticalLayout_6)
        self.vert_lay = QtWidgets.QVBoxLayout()
        self.vert_lay.setObjectName("vert_lay")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_3.setObjectName("label_3")
        self.vert_lay.addWidget(self.label_3)
        self.custom_models = QtWidgets.QVBoxLayout()
        self.custom_models.setObjectName("custom_models")
        self.vert_lay.addLayout(self.custom_models)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_lay.addItem(spacerItem4)
        self.horizontalLayout_2.addLayout(self.vert_lay)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem6)
        self.next = QtWidgets.QPushButton(self.centralwidget)
        self.next.setObjectName("next")
        self.verticalLayout_5.addWidget(self.next)
        self.horizontalLayout_2.addLayout(self.verticalLayout_5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        self.gridLayout_2.addLayout(self.verticalLayout_4, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "REAL Video Enhancer"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:22pt; font-weight:700;\">Select Rife Models to Download:</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Official models:"))
        self.rife.setText(_translate("MainWindow", "Rife (2X)"))
        self.rifeanime.setText(_translate("MainWindow", "Rife-Anime (2X)"))
        self.rifehd.setText(_translate("MainWindow", "Rife-HD (2X)"))
        self.rifeuhd.setText(_translate("MainWindow", "Rife-UHD (2X)"))
        self.rife2.setText(_translate("MainWindow", "Rife-V2 (2X)"))
        self.rife23.setText(_translate("MainWindow", "Rife-V2.3 (2X)"))
        self.rife24.setText(_translate("MainWindow", "Rife-V2.4 (2X)"))
        self.rife30.setText(_translate("MainWindow", "Rife-V3.0 (2X)"))
        self.rife31.setText(_translate("MainWindow", "Rife-V3.1 (2X)"))
        self.rife4.setText(_translate("MainWindow", "Rife-V4 (2X), (4X), (8X)"))
        self.rife46.setText(_translate("MainWindow", "Rife-V4.6 (2X), (4X), (8X) (recommended)"))
        self.rife47.setText(_translate("MainWindow", "Rife-V4.7 (2X), (4X), (8X) (recommended)"))
        self.rife48.setText(_translate("MainWindow", "Rife-V4.8 (2X), (4X), (8X) (recommended)"))
        self.label_3.setText(_translate("MainWindow", "Custom Models:"))
        self.next.setText(_translate("MainWindow", "Next"))

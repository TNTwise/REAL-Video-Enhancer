# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'testRVEInterface.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QProgressBar, QPushButton, QSizePolicy,
    QSpacerItem, QStackedWidget, QTextEdit, QVBoxLayout,
    QWidget)
import resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(897, 707)
        MainWindow.setStyleSheet(u"\n"
"\n"
"QMainWindow {\n"
"   color:black;\n"
"   background-color:black;\n"
"border:1px;\n"
"}\n"
"QLabel{\n"
"	color: #fff;\n"
"}\n"
"QLineEdit{\n"
"color: #fff;\n"
"}\n"
"#centralwidget{\n"
"	background-color:#1f232a;\n"
"}\n"
"#leftMenuSubContainer{\n"
"	background-color:#16191d;\n"
"   border-radius: 30px;\n"
"}\n"
"#bottomMenuSubContainer{\n"
"	background-color:#16191d;\n"
"   border-radius: 30px;\n"
"}\n"
"\n"
"QProgressBar{\n"
"    background-color:#2c313c;\n"
"	border-radius: 10px;\n"
"\n"
"}\n"
" QProgressBar::chunk {\n"
"     background-color:white;\n"
"	text-align:left;\n"
"	border-radius: 10px;\n"
"	\n"
" }\n"
"QStackedWidget{\n"
"	background-color:#16191d;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 25px;\n"
"}\n"
"QPlainTextEdit{\n"
"    background-color:#2c313c;\n"
"	border-radius: 30px;\n"
"\n"
"}\n"
"QPushButton{\n"
"    background-color:#2c313c;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 10px;\n"
"    color: #fff;\n"
"}\n"
"QPushButton:checked{\n"
""
                        "	background-color:#676e7b;\n"
"}\n"
"QPushButton:hover{\n"
"	background-color:#343b47;\n"
"}\n"
"\n"
"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.centralWidget = QWidget(self.centralwidget)
        self.centralWidget.setObjectName(u"centralWidget")
        self.centralWidget.setMinimumSize(QSize(0, 0))
        self.verticalLayout_2 = QVBoxLayout(self.centralWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.mainWindowContainer = QWidget(self.centralWidget)
        self.mainWindowContainer.setObjectName(u"mainWindowContainer")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainWindowContainer.sizePolicy().hasHeightForWidth())
        self.mainWindowContainer.setSizePolicy(sizePolicy)
        self.mainWindowContainer.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_3 = QHBoxLayout(self.mainWindowContainer)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.stackedWidget = QStackedWidget(self.mainWindowContainer)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setStyleSheet(u"")
        self.homePage = QWidget()
        self.homePage.setObjectName(u"homePage")
        self.homePage.setStyleSheet(u"QWidget{\n"
"background-color:#16191d\n"
"\n"
"}")
        self.gridLayout_2 = QGridLayout(self.homePage)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label = QLabel(self.homePage)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(25)
        self.label.setFont(font)

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.stackedWidget.addWidget(self.homePage)
        self.procPage = QWidget()
        self.procPage.setObjectName(u"procPage")
        self.procPage.setStyleSheet(u"QWidget{\n"
"background-color:#16191d\n"
"\n"
"}\n"
"QLineEdit{\n"
"background-color:#2c313c;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QTextEdit{\n"
"background-color:#2c313c;\n"
"border-radius: 10px;\n"
"padding:5px 10px;\n"
"}\n"
"\n"
"QPushButton{\n"
"    background-color:#2c313c;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:checked{\n"
"	background-color:#676e7b;\n"
"}\n"
"QPushButton:hover{\n"
"	background-color:#343b47;\n"
"}\n"
"\n"
"QComboBox{\n"
"background-color:#2c313c;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 10px;\n"
"color:white;\n"
"}\n"
"\n"
"QProgressBar{\n"
"    background-color:#2c313c;\n"
"	border-radius: 10px;\n"
"\n"
"}\n"
" QProgressBar::chunk {\n"
"     background-color:white;\n"
"	text-align:left;\n"
"	border-radius: 10px;\n"
"	\n"
" }")
        self.gridLayout_3 = QGridLayout(self.procPage)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.bottomMenuSubContainer = QWidget(self.procPage)
        self.bottomMenuSubContainer.setObjectName(u"bottomMenuSubContainer")
        self.bottomMenuSubContainer.setMinimumSize(QSize(0, 85))
        self.horizontalLayout_2 = QHBoxLayout(self.bottomMenuSubContainer)
        self.horizontalLayout_2.setSpacing(20)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(22, -1, 22, -1)
        self.startRenderButton = QPushButton(self.bottomMenuSubContainer)
        self.startRenderButton.setObjectName(u"startRenderButton")
        self.startRenderButton.setEnabled(True)
        self.startRenderButton.setMaximumSize(QSize(59, 16777215))
        icon = QIcon()
        icon.addFile(u":/icons/icons/play.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.startRenderButton.setIcon(icon)
        self.startRenderButton.setIconSize(QSize(55, 45))

        self.horizontalLayout_2.addWidget(self.startRenderButton)

        self.progressBar = QProgressBar(self.bottomMenuSubContainer)
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy1)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)

        self.horizontalLayout_2.addWidget(self.progressBar)


        self.gridLayout_3.addWidget(self.bottomMenuSubContainer, 1, 0, 1, 2)

        self.videoInfoContainer = QWidget(self.procPage)
        self.videoInfoContainer.setObjectName(u"videoInfoContainer")
        self.videoInfoContainer.setMaximumSize(QSize(16000, 16777215))
        self.videoInfoContainer.setStyleSheet(u"*:disabled{\n"
" \n"
"	color:gray;\n"
"}")
        self.verticalLayout_7 = QVBoxLayout(self.videoInfoContainer)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.inputFileContainer = QWidget(self.videoInfoContainer)
        self.inputFileContainer.setObjectName(u"inputFileContainer")
        self.horizontalLayout_4 = QHBoxLayout(self.inputFileContainer)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.inputFileSelectButton = QPushButton(self.inputFileContainer)
        self.inputFileSelectButton.setObjectName(u"inputFileSelectButton")
        self.inputFileSelectButton.setMinimumSize(QSize(0, 30))
        self.inputFileSelectButton.setMaximumSize(QSize(16777215, 30))
        font1 = QFont()
        font1.setPointSize(15)
        self.inputFileSelectButton.setFont(font1)
        self.inputFileSelectButton.setStyleSheet(u"")

        self.horizontalLayout_4.addWidget(self.inputFileSelectButton)

        self.inputFileText = QLineEdit(self.inputFileContainer)
        self.inputFileText.setObjectName(u"inputFileText")
        self.inputFileText.setMinimumSize(QSize(0, 30))

        self.horizontalLayout_4.addWidget(self.inputFileText)


        self.verticalLayout_7.addWidget(self.inputFileContainer)

        self.outputFileContainer = QWidget(self.videoInfoContainer)
        self.outputFileContainer.setObjectName(u"outputFileContainer")
        self.horizontalLayout_5 = QHBoxLayout(self.outputFileContainer)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.outputFileSelectButton = QPushButton(self.outputFileContainer)
        self.outputFileSelectButton.setObjectName(u"outputFileSelectButton")
        self.outputFileSelectButton.setEnabled(False)
        self.outputFileSelectButton.setMinimumSize(QSize(0, 30))
        self.outputFileSelectButton.setMaximumSize(QSize(16777215, 30))
        self.outputFileSelectButton.setFont(font1)
        self.outputFileSelectButton.setStyleSheet(u"")

        self.horizontalLayout_5.addWidget(self.outputFileSelectButton)

        self.outputFileText = QLineEdit(self.outputFileContainer)
        self.outputFileText.setObjectName(u"outputFileText")
        self.outputFileText.setEnabled(False)
        self.outputFileText.setMinimumSize(QSize(0, 30))

        self.horizontalLayout_5.addWidget(self.outputFileText)


        self.verticalLayout_7.addWidget(self.outputFileContainer)

        self.widget_3 = QWidget(self.videoInfoContainer)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_11 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_12 = QLabel(self.widget_3)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font1)

        self.horizontalLayout_11.addWidget(self.label_12)

        self.backendComboBox = QComboBox(self.widget_3)
        self.backendComboBox.setObjectName(u"backendComboBox")
        sizePolicy1.setHeightForWidth(self.backendComboBox.sizePolicy().hasHeightForWidth())
        self.backendComboBox.setSizePolicy(sizePolicy1)
        self.backendComboBox.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_11.addWidget(self.backendComboBox)


        self.verticalLayout_7.addWidget(self.widget_3)

        self.widget = QWidget(self.videoInfoContainer)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_7 = QLabel(self.widget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font1)

        self.horizontalLayout.addWidget(self.label_7)

        self.methodComboBox = QComboBox(self.widget)
        self.methodComboBox.addItem("")
        self.methodComboBox.addItem("")
        self.methodComboBox.setObjectName(u"methodComboBox")
        sizePolicy1.setHeightForWidth(self.methodComboBox.sizePolicy().hasHeightForWidth())
        self.methodComboBox.setSizePolicy(sizePolicy1)
        self.methodComboBox.setMinimumSize(QSize(250, 0))

        self.horizontalLayout.addWidget(self.methodComboBox)


        self.verticalLayout_7.addWidget(self.widget)

        self.widget_5 = QWidget(self.videoInfoContainer)
        self.widget_5.setObjectName(u"widget_5")
        self.horizontalLayout_13 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_14 = QLabel(self.widget_5)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setFont(font1)

        self.horizontalLayout_13.addWidget(self.label_14)

        self.modelComboBox = QComboBox(self.widget_5)
        self.modelComboBox.setObjectName(u"modelComboBox")
        sizePolicy1.setHeightForWidth(self.modelComboBox.sizePolicy().hasHeightForWidth())
        self.modelComboBox.setSizePolicy(sizePolicy1)
        self.modelComboBox.setMinimumSize(QSize(250, 0))

        self.horizontalLayout_13.addWidget(self.modelComboBox)


        self.verticalLayout_7.addWidget(self.widget_5)

        self.interpolationContainer = QWidget(self.videoInfoContainer)
        self.interpolationContainer.setObjectName(u"interpolationContainer")
        self.interpolationContainer.setEnabled(True)
        self.horizontalLayout_7 = QHBoxLayout(self.interpolationContainer)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_11 = QLabel(self.interpolationContainer)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font1)

        self.horizontalLayout_7.addWidget(self.label_11)

        self.interpolationMultiplierComboBox = QComboBox(self.interpolationContainer)
        self.interpolationMultiplierComboBox.addItem("")
        self.interpolationMultiplierComboBox.addItem("")
        self.interpolationMultiplierComboBox.addItem("")
        self.interpolationMultiplierComboBox.addItem("")
        self.interpolationMultiplierComboBox.addItem("")
        self.interpolationMultiplierComboBox.addItem("")
        self.interpolationMultiplierComboBox.addItem("")
        self.interpolationMultiplierComboBox.setObjectName(u"interpolationMultiplierComboBox")
        sizePolicy1.setHeightForWidth(self.interpolationMultiplierComboBox.sizePolicy().hasHeightForWidth())
        self.interpolationMultiplierComboBox.setSizePolicy(sizePolicy1)
        self.interpolationMultiplierComboBox.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_7.addWidget(self.interpolationMultiplierComboBox)


        self.verticalLayout_7.addWidget(self.interpolationContainer)

        self.renderOutput = QTextEdit(self.videoInfoContainer)
        self.renderOutput.setObjectName(u"renderOutput")
        sizePolicy.setHeightForWidth(self.renderOutput.sizePolicy().hasHeightForWidth())
        self.renderOutput.setSizePolicy(sizePolicy)
        self.renderOutput.setFont(font1)
        self.renderOutput.setStyleSheet(u"*:disabled{\n"
" \n"
"	color:white;\n"
"}")
        self.renderOutput.setReadOnly(True)

        self.verticalLayout_7.addWidget(self.renderOutput)


        self.gridLayout_3.addWidget(self.videoInfoContainer, 0, 0, 1, 1)

        self.processInfoContainer = QWidget(self.procPage)
        self.processInfoContainer.setObjectName(u"processInfoContainer")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.processInfoContainer.sizePolicy().hasHeightForWidth())
        self.processInfoContainer.setSizePolicy(sizePolicy2)
        self.verticalLayout_5 = QVBoxLayout(self.processInfoContainer)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.previewSubContainer = QWidget(self.processInfoContainer)
        self.previewSubContainer.setObjectName(u"previewSubContainer")
        self.verticalLayout_8 = QVBoxLayout(self.previewSubContainer)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.previewLabel = QLabel(self.previewSubContainer)
        self.previewLabel.setObjectName(u"previewLabel")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.previewLabel.sizePolicy().hasHeightForWidth())
        self.previewLabel.setSizePolicy(sizePolicy3)
        self.previewLabel.setStyleSheet(u"QLabel{\n"
"background-color:#2c313c;\n"
"border-radius: 10px;\n"
"}\n"
"")
        self.previewLabel.setScaledContents(True)

        self.verticalLayout_8.addWidget(self.previewLabel)


        self.verticalLayout_5.addWidget(self.previewSubContainer)

        self.infoSubContainer = QWidget(self.processInfoContainer)
        self.infoSubContainer.setObjectName(u"infoSubContainer")
        self.verticalLayout_9 = QVBoxLayout(self.infoSubContainer)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.videoInfoLabel = QLabel(self.infoSubContainer)
        self.videoInfoLabel.setObjectName(u"videoInfoLabel")
        sizePolicy2.setHeightForWidth(self.videoInfoLabel.sizePolicy().hasHeightForWidth())
        self.videoInfoLabel.setSizePolicy(sizePolicy2)
        self.videoInfoLabel.setStyleSheet(u"QLabel{\n"
"background-color:#2c313c;\n"
"border-radius: 10px;\n"
"}\n"
"")

        self.verticalLayout_9.addWidget(self.videoInfoLabel)


        self.verticalLayout_5.addWidget(self.infoSubContainer)


        self.gridLayout_3.addWidget(self.processInfoContainer, 0, 1, 1, 1)

        self.stackedWidget.addWidget(self.procPage)
        self.settingsPage = QWidget()
        self.settingsPage.setObjectName(u"settingsPage")
        self.settingsPage.setStyleSheet(u"QWidget{\n"
"background-color:#16191d\n"
"\n"
"}")
        self.gridLayout_4 = QGridLayout(self.settingsPage)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_3 = QLabel(self.settingsPage)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)

        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)

        self.stackedWidget.addWidget(self.settingsPage)
        self.downloadPage = QWidget()
        self.downloadPage.setObjectName(u"downloadPage")
        self.downloadPage.setStyleSheet(u"QWidget{\n"
"background-color:#16191d\n"
"\n"
"}\n"
"QCheckBox{\n"
"color:white;\n"
"\n"
"}\n"
"QLineEdit{\n"
"background-color:#2c313c;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton{\n"
"    background-color:#2c313c;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:checked{\n"
"	background-color:#676e7b;\n"
"}\n"
"QPushButton:hover{\n"
"	background-color:#343b47;\n"
"}\n"
"QPushButton:disabled{\n"
"	background-color:#343b47;\n"
"	color:gray;\n"
"}\n"
"QComboBox{\n"
"background-color:#2c313c;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 10px;\n"
"color:white;\n"
"}\n"
"Line{\n"
"color:white;\n"
"background-color:white;\n"
"}\n"
"\n"
"")
        self.gridLayout_5 = QGridLayout(self.downloadPage)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.backendSelectContainer = QWidget(self.downloadPage)
        self.backendSelectContainer.setObjectName(u"backendSelectContainer")
        self.verticalLayout_11 = QVBoxLayout(self.backendSelectContainer)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.label_4 = QLabel(self.backendSelectContainer)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)

        self.verticalLayout_11.addWidget(self.label_4)

        self.line_3 = QFrame(self.backendSelectContainer)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.Shape.VLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_11.addWidget(self.line_3)

        self.pytorchBackendInstallerContainer = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer.setObjectName(u"pytorchBackendInstallerContainer")
        self.horizontalLayout_6 = QHBoxLayout(self.pytorchBackendInstallerContainer)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.downloadTorchCUDABtn = QPushButton(self.pytorchBackendInstallerContainer)
        self.downloadTorchCUDABtn.setObjectName(u"downloadTorchCUDABtn")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.downloadTorchCUDABtn.sizePolicy().hasHeightForWidth())
        self.downloadTorchCUDABtn.setSizePolicy(sizePolicy4)
        self.downloadTorchCUDABtn.setMaximumSize(QSize(50, 16777215))
        icon1 = QIcon()
        icon1.addFile(u":/icons/icons/download.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.downloadTorchCUDABtn.setIcon(icon1)
        self.downloadTorchCUDABtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_6.addWidget(self.downloadTorchCUDABtn)

        self.label_6 = QLabel(self.pytorchBackendInstallerContainer)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_6.addWidget(self.label_6)


        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer)

        self.pytorchBackendInstallerContainer_2 = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer_2.setObjectName(u"pytorchBackendInstallerContainer_2")
        self.horizontalLayout_8 = QHBoxLayout(self.pytorchBackendInstallerContainer_2)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.downloadTensorRTBtn = QPushButton(self.pytorchBackendInstallerContainer_2)
        self.downloadTensorRTBtn.setObjectName(u"downloadTensorRTBtn")
        sizePolicy4.setHeightForWidth(self.downloadTensorRTBtn.sizePolicy().hasHeightForWidth())
        self.downloadTensorRTBtn.setSizePolicy(sizePolicy4)
        self.downloadTensorRTBtn.setMaximumSize(QSize(50, 16777215))
        self.downloadTensorRTBtn.setIcon(icon1)
        self.downloadTensorRTBtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_8.addWidget(self.downloadTensorRTBtn)

        self.label_8 = QLabel(self.pytorchBackendInstallerContainer_2)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_8.addWidget(self.label_8)


        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer_2)

        self.pytorchBackendInstallerContainer_3 = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer_3.setObjectName(u"pytorchBackendInstallerContainer_3")
        self.horizontalLayout_9 = QHBoxLayout(self.pytorchBackendInstallerContainer_3)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.downloadTorchROCmBtn = QPushButton(self.pytorchBackendInstallerContainer_3)
        self.downloadTorchROCmBtn.setObjectName(u"downloadTorchROCmBtn")
        sizePolicy4.setHeightForWidth(self.downloadTorchROCmBtn.sizePolicy().hasHeightForWidth())
        self.downloadTorchROCmBtn.setSizePolicy(sizePolicy4)
        self.downloadTorchROCmBtn.setMaximumSize(QSize(50, 16777215))
        self.downloadTorchROCmBtn.setIcon(icon1)
        self.downloadTorchROCmBtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_9.addWidget(self.downloadTorchROCmBtn)

        self.label_9 = QLabel(self.pytorchBackendInstallerContainer_3)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_9.addWidget(self.label_9)


        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer_3)

        self.pytorchBackendInstallerContainer_4 = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer_4.setObjectName(u"pytorchBackendInstallerContainer_4")
        self.horizontalLayout_10 = QHBoxLayout(self.pytorchBackendInstallerContainer_4)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.downloadNCNNBtn = QPushButton(self.pytorchBackendInstallerContainer_4)
        self.downloadNCNNBtn.setObjectName(u"downloadNCNNBtn")
        sizePolicy4.setHeightForWidth(self.downloadNCNNBtn.sizePolicy().hasHeightForWidth())
        self.downloadNCNNBtn.setSizePolicy(sizePolicy4)
        self.downloadNCNNBtn.setMaximumSize(QSize(50, 16777215))
        self.downloadNCNNBtn.setIcon(icon1)
        self.downloadNCNNBtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_10.addWidget(self.downloadNCNNBtn)

        self.label_10 = QLabel(self.pytorchBackendInstallerContainer_4)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_10.addWidget(self.label_10)


        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer_4)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_4)


        self.gridLayout_5.addWidget(self.backendSelectContainer, 0, 0, 1, 1)

        self.modelSelectContainer = QWidget(self.downloadPage)
        self.modelSelectContainer.setObjectName(u"modelSelectContainer")
        self.verticalLayout_10 = QVBoxLayout(self.modelSelectContainer)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.label_5 = QLabel(self.modelSelectContainer)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout_10.addWidget(self.label_5)

        self.line = QFrame(self.modelSelectContainer)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.VLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_10.addWidget(self.line)

        self.spanCheckBox = QCheckBox(self.modelSelectContainer)
        self.spanCheckBox.setObjectName(u"spanCheckBox")
        self.spanCheckBox.setChecked(False)

        self.verticalLayout_10.addWidget(self.spanCheckBox)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_10.addItem(self.verticalSpacer_3)

        self.label_2 = QLabel(self.modelSelectContainer)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_10.addWidget(self.label_2)

        self.line_2 = QFrame(self.modelSelectContainer)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.VLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_10.addWidget(self.line_2)

        self.checkBox = QCheckBox(self.modelSelectContainer)
        self.checkBox.setObjectName(u"checkBox")

        self.verticalLayout_10.addWidget(self.checkBox)

        self.downloadModelsButton = QPushButton(self.modelSelectContainer)
        self.downloadModelsButton.setObjectName(u"downloadModelsButton")

        self.verticalLayout_10.addWidget(self.downloadModelsButton)


        self.gridLayout_5.addWidget(self.modelSelectContainer, 0, 1, 1, 1)

        self.stackedWidget.addWidget(self.downloadPage)

        self.horizontalLayout_3.addWidget(self.stackedWidget)


        self.verticalLayout_2.addWidget(self.mainWindowContainer)


        self.gridLayout.addWidget(self.centralWidget, 0, 1, 1, 1)

        self.leftMenuContainer = QWidget(self.centralwidget)
        self.leftMenuContainer.setObjectName(u"leftMenuContainer")
        self.leftMenuContainer.setStyleSheet(u"")
        self.verticalLayout = QVBoxLayout(self.leftMenuContainer)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(9, -1, 0, -1)
        self.leftMenuSubContainer = QWidget(self.leftMenuContainer)
        self.leftMenuSubContainer.setObjectName(u"leftMenuSubContainer")
        self.verticalLayout_3 = QVBoxLayout(self.leftMenuSubContainer)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, -1, 9, -1)
        self.verticalWidget_2 = QWidget(self.leftMenuSubContainer)
        self.verticalWidget_2.setObjectName(u"verticalWidget_2")
        self.verticalLayout_6 = QVBoxLayout(self.verticalWidget_2)
        self.verticalLayout_6.setSpacing(9)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(9, 9, 9, 9)
        self.homeBtn = QPushButton(self.verticalWidget_2)
        self.homeBtn.setObjectName(u"homeBtn")
        self.homeBtn.setMinimumSize(QSize(31, 0))
        self.homeBtn.setMaximumSize(QSize(55, 16777215))
        icon2 = QIcon()
        icon2.addFile(u":/icons/icons/home.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.homeBtn.setIcon(icon2)
        self.homeBtn.setIconSize(QSize(35, 35))
        self.homeBtn.setCheckable(True)
        self.homeBtn.setChecked(True)

        self.verticalLayout_6.addWidget(self.homeBtn)

        self.processBtn = QPushButton(self.verticalWidget_2)
        self.processBtn.setObjectName(u"processBtn")
        self.processBtn.setMaximumSize(QSize(55, 16777215))
        icon3 = QIcon()
        icon3.addFile(u":/icons/icons/cpu.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.processBtn.setIcon(icon3)
        self.processBtn.setIconSize(QSize(35, 35))
        self.processBtn.setCheckable(True)

        self.verticalLayout_6.addWidget(self.processBtn)


        self.verticalLayout_3.addWidget(self.verticalWidget_2, 0, Qt.AlignmentFlag.AlignLeft)

        self.verticalSpacer = QSpacerItem(20, 1000000, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.verticalWidget = QWidget(self.leftMenuSubContainer)
        self.verticalWidget.setObjectName(u"verticalWidget")
        self.verticalLayout_4 = QVBoxLayout(self.verticalWidget)
        self.verticalLayout_4.setSpacing(9)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(-1, -1, 9, -1)
        self.settingsBtn = QPushButton(self.verticalWidget)
        self.settingsBtn.setObjectName(u"settingsBtn")
        sizePolicy4.setHeightForWidth(self.settingsBtn.sizePolicy().hasHeightForWidth())
        self.settingsBtn.setSizePolicy(sizePolicy4)
        self.settingsBtn.setMaximumSize(QSize(55, 16777215))
        self.settingsBtn.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.settingsBtn.setStyleSheet(u"text-align:left;\n"
"\n"
"")
        icon4 = QIcon()
        icon4.addFile(u":/icons/icons/settings.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.settingsBtn.setIcon(icon4)
        self.settingsBtn.setIconSize(QSize(35, 35))
        self.settingsBtn.setCheckable(True)

        self.verticalLayout_4.addWidget(self.settingsBtn)

        self.downloadBtn = QPushButton(self.verticalWidget)
        self.downloadBtn.setObjectName(u"downloadBtn")
        self.downloadBtn.setMaximumSize(QSize(55, 16777215))
        self.downloadBtn.setIcon(icon1)
        self.downloadBtn.setIconSize(QSize(35, 35))
        self.downloadBtn.setCheckable(True)

        self.verticalLayout_4.addWidget(self.downloadBtn)

        self.githubBtn = QPushButton(self.verticalWidget)
        self.githubBtn.setObjectName(u"githubBtn")
        self.githubBtn.setMaximumSize(QSize(55, 16777215))
        icon5 = QIcon()
        icon5.addFile(u":/icons/icons/github.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.githubBtn.setIcon(icon5)
        self.githubBtn.setIconSize(QSize(35, 35))
        self.githubBtn.setCheckable(False)

        self.verticalLayout_4.addWidget(self.githubBtn)


        self.verticalLayout_3.addWidget(self.verticalWidget, 0, Qt.AlignmentFlag.AlignLeft)


        self.verticalLayout.addWidget(self.leftMenuSubContainer, 0, Qt.AlignmentFlag.AlignLeft)


        self.gridLayout.addWidget(self.leftMenuContainer, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Home", None))
        self.startRenderButton.setText("")
        self.inputFileSelectButton.setText(QCoreApplication.translate("MainWindow", u"Select Input File", None))
        self.outputFileSelectButton.setText(QCoreApplication.translate("MainWindow", u"Select Output Folder", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Backend", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Method", None))
        self.methodComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Interpolate", None))
        self.methodComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Upscale", None))

        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Model", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Interpolation Multiplier", None))
        self.interpolationMultiplierComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"2", None))
        self.interpolationMultiplierComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"3", None))
        self.interpolationMultiplierComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"4", None))
        self.interpolationMultiplierComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"5", None))
        self.interpolationMultiplierComboBox.setItemText(4, QCoreApplication.translate("MainWindow", u"6", None))
        self.interpolationMultiplierComboBox.setItemText(5, QCoreApplication.translate("MainWindow", u"7", None))
        self.interpolationMultiplierComboBox.setItemText(6, QCoreApplication.translate("MainWindow", u"8", None))

        self.renderOutput.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:15pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'Sans Serif';\"><br /></p></body></html>", None))
        self.previewLabel.setText("")
        self.videoInfoLabel.setText(QCoreApplication.translate("MainWindow", u"videoInfo", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Settngs", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Backends", None))
        self.downloadTorchCUDABtn.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"PyTorch CUDA (Nvidia Only) ", None))
        self.downloadTensorRTBtn.setText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"TensorRT (Nvidia RTX 20 series and up)", None))
        self.downloadTorchROCmBtn.setText("")
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"PyTorch ROCm (AMD Unknown Compatiblity)", None))
        self.downloadNCNNBtn.setText("")
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"NCNN Vulkan (All GPUs, Slower)", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Upscaling Models", None))
        self.spanCheckBox.setText(QCoreApplication.translate("MainWindow", u"SPAN", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Interpolation Models", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"RIFE", None))
        self.downloadModelsButton.setText(QCoreApplication.translate("MainWindow", u"Download", None))
        self.homeBtn.setText("")
        self.processBtn.setText("")
        self.settingsBtn.setText("")
        self.downloadBtn.setText("")
    # retranslateUi


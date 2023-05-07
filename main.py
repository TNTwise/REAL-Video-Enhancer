from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox

import mainwindow
import os
from threading import *
from src.settings import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.start as start
import src.get_rife_models as get_rife_models 
import src.get_realsr_models as get_realsr_models



thisdir = os.getcwd()
homedir = os.path.expanduser(r"~")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        #Define Variables
        self.input_file = ''
        self.output_folder = ''
        self.output_folder = settings.OutputDir 

        self.pin_functions()
        self.show()

    def pin_functions(self):

        self.ui.Input_video_rife.clicked.connect(self.openFileNameDialog)
        self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)
        self.ui.Rife_Model.setCurrentIndex(5)
        self.ui.RifeStart.clicked.connect(self.startRife)

    def openFileNameDialog(self):

        self.input_file = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]

    def openFolderDialog(self):
        
        self.output_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
    
    def _setStyle(self,color):
        self.ui.RifeStart.setStyleSheet(f'color: {color};')
        self.ui.Input_video_rife.setStyleSheet(f'color: {color};')
        self.ui.Output_folder_rife.setStyleSheet(f'color: {color};')
        self.ui.Rife_Model.setStyleSheet(f"QComboBox {{ color: {color}; }}")
        self.ui.Rife_Times.setStyleSheet(f"QComboBox {{ color: {color}; }}")

    def setDisableEnable(self,mode):
        self.ui.RifeStart.setDisabled(mode)
        self.ui.Input_video_rife.setDisabled(mode)
        self.ui.Output_folder_rife.setDisabled(mode)
        self.ui.Rife_Model.setDisabled(mode)
        self.ui.Rife_Times.setDisabled(mode)

    def endRife(self):
        self.rifeThread.join()
        
        current_palette = app.style().standardPalette()
        self.setDisableEnable(False)
        if current_palette.color(current_palette.WindowText).lightness() > 127:
            self._setStyle('white')
        else:
            self._setStyle('black')

    def startRife(self):

        if self.input_file != '':
            self.setDisableEnable(True)
            self._setStyle('gray')
            self.rifeThread = Thread(target=lambda: start.start_rife((self.ui.Rife_Model.currentText().lower()),int(self.ui.Rife_Times.currentText()[0]),self.input_file,self.output_folder))
            self.rifeThread.start()
            Thread(target=self.endRife).start()
        else:
            self.showDialogBox("No input file selected.")

    

    def showDialogBox(self,message):
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        msg.setText(f"{message}")
        msg.exec_()
    


settings = Settings()


    
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
sys.exit(app.exec_())
    


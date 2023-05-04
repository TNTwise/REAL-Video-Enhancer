from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox

import mainwindow
import os
from threading import *
import src.start as start
import src.get_models as get_models
from src.settings import *
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
    def endRife(self):
        self.rifeThread.join()
        self.ui.RifeStart.setDisabled(False)
        self.ui.RifeStart.setStyleSheet('color: white;')
    def startRife(self):

        if self.input_file != '':
            self.ui.RifeStart.setEnabled(False)
            self.ui.RifeStart.setStyleSheet('color: gray;')
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

if __name__ == '__main__':
    settings = Settings()
    
    if os.path.exists(f'{thisdir}/Real-ESRGAN/') == False or os.path.exists(f"{thisdir}/rife-vulkan-models/") == False:
        
        get_models.get_all_models()
        
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

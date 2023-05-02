from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import mainwindow
import os
thisdir = os.getcwd()
homedir = os.path.expanduser(r"~")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        
        
        ui = mainwindow.Ui_MainWindow()
        ui.setupUi(self)

        ui.Input_video_rife.clicked.connect(self.openFileNameDialog)
        self.show()
    def openFileNameDialog(self):
        
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
    f'{homedir}',"Video files (*.mp4)")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

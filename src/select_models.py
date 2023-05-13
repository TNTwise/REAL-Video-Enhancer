# WORK IN PROGRESS!!!!

import os
thisdir = os.getcwd()
import sys
import requests
import re
from zipfile import ZipFile
from PyQt5 import QtWidgets, uic
from time import sleep
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout, QMessageBox
from src.settings import *
from src.return_data import *
from threading import Thread
import src.SelectModels as SelectModels

if os.path.exists(f'{thisdir}/Real-ESRGAN/') == False or os.path.exists(f'{thisdir}/rife-vulkan-models/') == False:

    class ChooseModels(QtWidgets.QMainWindow):
            def __init__(self):
                super(ChooseModels, self).__init__()
                self.ui = SelectModels.Ui_MainWindow()
                self.ui.setupUi(self)
                self.pinFunctions()
                self.show()
            def pinFunctions(self):
                
                self.ui.next.clicked.connect(self.nextfunction)
            def nextfunction(self):
                
                self.close()
                import src.get_models as get_models
                
                


    app = QtWidgets.QApplication(sys.argv)
    window = ChooseModels()
    sys.exit(app.exec_())
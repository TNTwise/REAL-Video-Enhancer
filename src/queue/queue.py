import src.getModels.select_models as sel_mod
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import cv2
import psutil
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QTextCursor, QPixmap,QIcon, QIntValidator
import PyQt5.QtCore as QtCore
import mainwindow
import os
from threading import *
from src.settings import *
from src.return_data import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.runAI.start as start
import src.workers as workers
import time
#import src.get_models as get_models
from time import sleep
import src.getModels.get_models as get_models
from multiprocessing import cpu_count
from src.messages import *
import modules.Rife as rife
import modules.ESRGAN as esrgan
import pypresence
import src.onProgramStart
def addToQueue(self):
    self.queueFile = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]
    self.QueueList.append(self.queueFile)
    self.queueVideoName = VideoName.return_video_name(self.queueFile)
    self.ui.QueueListWidget.addItem(self.queueVideoName)
    self.ui.QueueListWidget.show()
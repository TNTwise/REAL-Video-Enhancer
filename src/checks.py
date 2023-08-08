import os
import requests
import sys
import src.thisdir
thisdir = src.thisdir.thisdir()
from zipfile import ZipFile
from PyQt5 import QtWidgets, uic
from time import sleep
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout, QMessageBox
from src.settings import *
from src.return_data import *
from threading import Thread
def check_if_models_exist(thisdir):
    if os.path.exists(f'{thisdir}/models/'):
        return True
    else:
        return False
    
def check_if_online():
    online=False
    try:
        requests.get('https://raw.githubusercontent.com/')
        online=True
    except:
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        msg.setText(f"You are offline, please connect to the internet to download the models.")
        sys.exit(msg.exec_())
    return online

def check_if_enough_space(render,input_file):
    pass
import os
def check_if_models_exist(thisdir):
    if os.path.exists(f'{thisdir}/models/'):
        return True
    else:
        return False
import requests
import sys
import os
import src.thisdir
thisdir = src.thisdir.thisdir()
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
def check_if_online():
    online=False
    try:
        requests.get('https://raw.githubusercontent.com/')
        online=True
    except:
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        msg.setText(f"You are offline, please connect to the internet to download the models or download the offline binary.")
        sys.exit(msg.exec_())
    return online
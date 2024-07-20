import os


def build_gui():
    os.system("pyside6-uic -g python testRVEInterface.ui > mainwindow.py")


def build_resources():
    os.system("pyside6-rcc resources.qrc > resources_rc.py")


build_gui()
build_resources()

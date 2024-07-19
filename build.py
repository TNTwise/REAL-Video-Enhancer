from PyQt6 import sip
from PyQt6.uic import compileUi
import os


def build_gui():
    with open("mainwindow.py", "w") as f:
        compileUi("testRVEInterface.ui", f)
        f.write("\nimport resources_rc")


def build_resources():
    os.system("pyrcc5 resources.qrc > resources_rc.py")
    with open("resources_rc.py", "r") as f:
        lines = f.readlines()

    with open("resources_rc.py", "w") as f:
        for i in lines:
            i = i.replace("PyQt5", "PyQt6")
            f.write(i)


build_gui()
build_resources()

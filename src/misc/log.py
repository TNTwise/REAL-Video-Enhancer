import datetime
import os
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QMessageBox,
)
from src.misc.createDirectories import createDirectories

current_time = datetime.datetime.today().strftime("%Y-%m-%d:%H:%M:%S")
from src.programData.thisdir import thisdir

thisdir = thisdir()
try:
    createDirectories()

except:
    if len(os.listdir(f"{thisdir}/logs/")) > 4:
        oldest_file = min(
            os.listdir(f"{thisdir}/logs/"),
            key=lambda x: os.path.getctime(os.path.join(f"{thisdir}/logs/", x)),
        )
        os.remove(f"{thisdir}/logs/{oldest_file}")
error_count = []


def log(log):
    if log in error_count:
        # Increment the count if the error has occurred before
        pass

    else:
        # Initialize count if the error is encountered for the first time
        error_count.append(log)

        with open(f"{thisdir}/logs/log_{current_time}.txt", "a") as f:
            f.write(str(log) + "\n")


class PopupWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Logs")
        self.setGeometry(200, 200, 400, 300)

        # Create a tab widget
        tab_widget = QTabWidget(self)

        # Define the number of tabs

        # Create tabs in a loop
        files = sorted(os.listdir(f"{thisdir}/logs/"))
        files.reverse()

        for i in range(len(files)):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)

            self.text_edit = QPlainTextEdit()
            self.text_edit.setReadOnly(True)

            self.text_edit.setMinimumSize(600, 600)
            # Add some text to the textbox

            log_file = f"{thisdir}/logs/{files[i]}"

            with open(log_file, "r") as f:
                log_list = f.readlines()
            log_string = ""
            for j in log_list:
                log_string += f"{j}"
            self.text_edit.setPlainText(log_string)

            # define buttons
            close_button = QPushButton("Close", self)
            close_button.clicked.connect(self.close)

            copy_button = QPushButton("Copy to Clipboard")
            copy_button.clicked.connect(self.copy_to_clipboard)

            # layout in order
            tab_layout.addWidget(self.text_edit)
            if "FLATPAK_ID" not in os.environ:
                tab_layout.addWidget(copy_button)
            tab_layout.addWidget(close_button)

            tab_widget.addTab(tab, f"{files[i]}")
            # Set the layout for the dialog

        # Create a layout for the dialog and add the tab widget
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(tab_widget)

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        QMessageBox.information(self, "Information", "Text copied to clipboard.")


def viewLogs(self):
    popup = PopupWindow()
    popup.exec_()

import src.thisdir
import datetime
import os
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLineEdit, QPushButton, QPlainTextEdit, QMessageBox
current_time = datetime.datetime.now()

thisdir = src.thisdir.thisdir()
try:
    os.mkdir(f'{thisdir}/logs/')
    
except:
        if len(os.listdir(f'{thisdir}/logs/')) > 4:
            oldest_file = min(os.listdir(f'{thisdir}/logs/'), key=lambda x: os.path.getctime(os.path.join(f'{thisdir}/logs/', x)))
            os.remove(f'{thisdir}/logs/{oldest_file}')
            
def log(log):
    last_line_in_log_file = ''
    if os.path.isfile(f'{thisdir}/logs/log_{current_time}.txt'):
        with open(f'{thisdir}/logs/log_{current_time}.txt', 'r') as f:
            last_line_in_log_file = f.readlines()[-1]
            print((last_line_in_log_file))
    

    with open(f'{thisdir}/logs/log_{current_time}.txt', 'a') as f:
        if last_line_in_log_file != log:
            f.write(str(log) + '\n')

class PopupWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Log')

        layout = QVBoxLayout()

        # Create an uneditable textbox
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)

        self.text_edit.setMinimumSize(600,600)
        # Add some text to the textbox
        log = sorted(os.listdir(f'{thisdir}/logs/'))[-1]

        log_file = f'{thisdir}/logs/{log}'

        with open(log_file, 'r') as f:
            log_list = f.readlines()
        log_string=''
        for i in log_list:
            log_string+=f'{i}'
        self.text_edit.setPlainText(log_string)


        #define buttons
        close_button = QPushButton('Close', self)
        close_button.clicked.connect(self.close)

        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)

        #layout in order
        layout.addWidget(self.text_edit)
        if 'FLATPAK_ID' not in os.environ:
            layout.addWidget(copy_button)
        layout.addWidget(close_button)

        # Set the layout for the dialog
        self.setLayout(layout)
    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        QMessageBox.information(self, "Information", "Text copied to clipboard.")

def viewLogs(self):
    popup = PopupWindow()
    popup.exec_()


    

    
    


    
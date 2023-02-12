import re
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import requests
import os 
import threading
from install_dependencies import dependencies

class wget:
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename
    def download(self):
        response = requests.get(self.url)
        total = int(response.headers.get('content-length', 0))
        with open(f"{self.filename}", 'wb') as f:
            for data in response.iter_content(1024):
                # Update progress bar
                f.write(data)
    



class InstallerWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QVBoxLayout())
        self.setWindowTitle("Installer")
        self.UI()
        self.setFixedSize(280, 200) 
        #Show the app
        self.show()

    def UI(self):
        
        
        self.is_dynamic_checkbox = qtw.QCheckBox("Install models when needed.", self)
        
        self.layout().addWidget(self.is_dynamic_checkbox)
        self.is_dynamic_checkbox.setGeometry(20,10,200,20)
        self.install_all_checkbox = qtw.QCheckBox("Install all models now.", self)
        self.is_dynamic_checkbox.stateChanged.connect(self.next_button_dynamic)

        
        
        self.layout().addWidget(self.install_all_checkbox)
        self.install_all_checkbox.setGeometry(20,40,200,20)
        
        self.install_all_checkbox.stateChanged.connect(self.next_button_install)

    def next_button_install(self):
        
        self.is_dynamic_checkbox.setChecked(False)
        
        # Create new next button
        self.next_install_button = qtw.QPushButton("Next", self)
        self.layout().addWidget(self.next_install_button)
        self.next_install_button.setGeometry(200,150,60,40)

        #try:
        #    self.next_dynamic_button
    def next_button_dynamic(self):
        self.install_all_checkbox.setChecked(False)
        self.next_dynamic_button = qtw.QPushButton("Next", self)
        self.layout().addWidget(self.next_dynamic_button)
        self.next_dynamic_button.setGeometry(200,150,60,40)
        self.next_dynamic_button.clicked.connect(self.next_command)
    def next_command(self):
        dependencies.install_dependencies()
app = qtw.QApplication([])
iw = InstallerWindow()
app.exec_()

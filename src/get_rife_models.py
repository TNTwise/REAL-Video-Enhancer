import os
thisdir = os.getcwd()
import sys
import requests
import re
import zipfile
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout, QMessageBox
from src.settings import *
from src.return_data import *
from threading import Thread
import src.get_realsr_models as get_realsr_models
class PopUpProgressB(QWidget):
    def __init__(self):


        super(PopUpProgressB, self).__init__()

        self.progressBarRife()



    def progressBarRife(self):

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 75)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.pbar)
        self.setLayout(self.layout)
        self.setGeometry(300, 300, 550, 100)
        self.setWindowTitle(f'Downloading Rife Models')
        self.show()

        Thread(target=self.show_loading_window).start()



    def latest_rife(self):
        latest = requests.get('https://github.com/nihui/rife-ncnn-vulkan/releases/latest/')
        latest = latest.url
        latest = re.findall(r'[\d]*$', latest)
        latest = latest[0]
        return(latest)

    def show_loading_window(self):

        os.chdir(f"{thisdir}/files/")

        version = self.latest_rife() # calls latest function which gets the latest version release of rife and returns the latest and the current, if the version file doesnt exist, it updates and creates the file
        latest_ver = version
        try:

            file=f"rife-ncnn-vulkan-{latest_ver}-ubuntu.zip"
            response = requests.get(f"https://github.com/nihui/rife-ncnn-vulkan/releases/download/{latest_ver}/rife-ncnn-vulkan-{latest_ver}-ubuntu.zip", stream=True)

            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte

            self.pbar.setMaximum(total_size_in_bytes)

            total_block_size = 0
            with open(file, 'wb') as f:
                for data in response.iter_content(block_size):
                    total_block_size += block_size
                    self.pbar.setValue(total_block_size)

                    f.write(data)

                Thread(target=self.get_rife).start()

                self.close()

        except:
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            msg.setText(f"You are offline, please download the offline binary or reconnect to the internet.")
            msg.exec_()


    def get_rife(self):

            version = self.latest_rife() # calls latest function which gets the latest version release of rife and returns the latest and the current, if the version file doesnt exist, it updates and creates the file
            latest_ver = version


            os.chdir(f"{thisdir}")
            with zipfile.ZipFile(f'{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu.zip', 'r') as zip_ref:
                zip_ref.extractall(f'{thisdir}/files/')
            os.system(f'mkdir -p "{thisdir}/rife-vulkan-models"')
            os.system(f'mv "{thisdir}/rife-ncnn-vulkan-{latest_ver}-ubuntu" "{thisdir}/files/"')
            os.system(f'mv "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu/"* "{thisdir}/rife-vulkan-models/"')
            Settings.change_setting(0,'rifeversion', f'{latest_ver}')
            os.system(f'rm -rf "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu.zip"')
            os.system(f'rm -rf "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu"')
            os.system(f'chmod +x "{thisdir}/rife-vulkan-models/rife-ncnn-vulkan"')

            if os.path.exists(f'{thisdir}/Real-ESRGAN/') == False:
                get_realsr_models.StartRealSR()
            else:
                import main as main
class StartRife:
    if ManageFiles.isfolder(f"{thisdir}/rife-vulkan-models") == False:
        app = QApplication(sys.argv)
        main_window = PopUpProgressB()

        sys.exit(app.exec_())

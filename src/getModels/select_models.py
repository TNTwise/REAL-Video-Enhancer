# WORK IN PROGRESS!!!!

import os
import sys
import requests
import re
import src.programData.thisdir

thisdir = src.programData.thisdir.thisdir()
if os.path.exists(f"{thisdir}/renders/") == False:
    os.mkdir(f"{thisdir}/renders/")
from zipfile import ZipFile
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QIcon
import src.misc.messages
import src.getModels.SelectModels as SelectModels
import src.getModels.Download as DownloadUI
from src.misc.log import log

global rife_install_list
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
from src.misc.messages import *
from src.programData.checks import *
import tarfile
from sys import exit
from src.getModels.rifeModelsFunctions import *
from src.programData.settings import *

settings = Settings()
from src.getModels.returnModelList import *

import src.getModels.googleDriveDownload as GDrive


def handleCUDAModels(model: str = ""):
    if "rife" and "pkl" in model:
        os.system(
            f'mkdir -p "{thisdir}/models/rife-cuda/{model.replace(".","").replace("pkl","")}" '
        )
        os.system(
            f'cp "{thisdir}/files/{model}" "{thisdir}/models/rife-cuda/{model.replace(".","").replace("pkl","")}" '
        )



import src.getModels.SelectAI as SelectAI
import traceback


class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(list)

    def __init__(self, parent):
        self.main = parent
        QThread.__init__(self, None)

    @pyqtSlot()
    def install_modules(self):
        settings = Settings()
        try:
            with open(f"{thisdir}/models.txt", "w") as f:
                f.write("")

            total_size_in_bytes = 0
            data_downloaded = 0
            try:  # this is due to the error message install_modules_dict showing up after exiting on fist screen, if i add this it stops it(very good programming)
                for link, name in install_modules_dict.items():
                    response = requests.get(link, stream=True)
                    total_size_in_bytes += int(
                        response.headers.get("content-length", 0)
                    )
            except:
                log("Not Installing!")
                exit()
            if check_if_enough_space_for_install(total_size_in_bytes) == False:
                return 0
            for link, name in install_modules_dict.items():
                response = requests.get(link, stream=True)
                with open(f"{thisdir}/files/{name}", "wb") as f:
                    for data in response.iter_content(1024):
                        f.write(data)
                        data_downloaded += 1024
                        self.intReady.emit(
                            [int(data_downloaded), total_size_in_bytes]
                        )  # sends back data to main thread# sends back data to main thread
            if os.path.exists(f"{settings.ModelDir}") == False:
                os.mkdir(f"{settings.ModelDir}")

            for i in os.listdir(f"{thisdir}/files/"):
                if os.path.exists(f"{thisdir}/bin/") == False:
                    os.mkdir(f"{thisdir}/bin/")
                if i == "ffmpeg":
                    os.system(f'chmod +x "{thisdir}/files/ffmpeg"')
                    os.system(f'mv "{thisdir}/files/ffmpeg" "{thisdir}/bin/"')
                if i == "yt-dlp_linux":
                    os.system(f'chmod +x "{thisdir}/files/yt-dlp_linux"')
                    os.system(f'mv "{thisdir}/files/yt-dlp_linux" "{thisdir}/bin/"')
                if i == "glxinfo":
                    os.system(f'chmod +x "{thisdir}/files/glxinfo"')
                    os.system(f'mv "{thisdir}/files/glxinfo" "{thisdir}/bin/"')
                if i == "rife-ncnn-vulkan":
                    try:
                        os.mkdir(f"{settings.ModelDir}/rife/")
                    except:
                        pass
                    os.system(f'chmod +x "{thisdir}/files/rife-ncnn-vulkan"')
                    os.system(
                        f'mv "{thisdir}/files/rife-ncnn-vulkan" "{thisdir}/models/rife/"'
                    )

                if ".zip" in i:
                    with ZipFile(f"{thisdir}/files/{i}", "r") as zip_ref:
                        zip_ref.extractall(f"{thisdir}/files/")
                        name = i.replace(".zip", "")
                        if return_data.returnOperatingSystem() == "MacOS":
                            name = name.replace("ubuntu", "macos")
                        if "-ncnn-vulkan" in name:
                            original_ai_name_ncnn_vulkan = re.findall(
                                r"[\w]*-ncnn-vulkan", name
                            )[0]
                            original_ai_name = original_ai_name_ncnn_vulkan.replace(
                                "-ncnn-vulkan", ""
                            )
                        else:
                            original_ai_name = name
                            original_ai_name_ncnn_vulkan = name

                    os.system(
                        f'mv "{thisdir}/files/{name}" "{settings.ModelDir}/{original_ai_name}"'
                    )
                    os.system(
                        f'chmod +x "{settings.ModelDir}/{original_ai_name}/{original_ai_name_ncnn_vulkan}"'
                    )

                if ".tar.gz" in i:
                    with tarfile.open(f"{thisdir}/files/{i}", "r") as f:
                        f.extractall(f"{settings.ModelDir}/rife/")
                handleCUDAModels(i)
            for i in os.listdir(f"{thisdir}/files/"):
                if ".txt" not in i:
                    os.remove(f"{thisdir}/files/{i}")
            self.finished.emit()
        except Exception as e:
            traceback_info = traceback.format_exc()
            log(f"ERROR: {e} {traceback_info}")
            self.main.showDialogBox(e)


def clear_files():
    for i in os.listdir(f"{thisdir}/files/"):
        if ".txt" not in i:
            try:
                os.remove(f"{thisdir}/files/{i}")
            except:
                os.system(f'rm -rf "{thisdir}/files/{i}"')


def install_icons(self):
    if os.path.exists(f"{thisdir}/icons/") == False:
        if check_if_online(True):
            try:
                log("Downloaded Icons")
                url = "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/icons.zip"
                local_filename = url.split("/")[-1]
                r = requests.get(url)
                f = open(f"{thisdir}/{local_filename}", "wb")
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                f.close()
                with ZipFile(f"{thisdir}/{local_filename}", "r") as f:
                    f.extractall(path=f"{thisdir}/")
                os.remove(f"{thisdir}/{local_filename}")
            except Exception as e:
                traceback_log = traceback.format_exc()
                log(f"ERROR: {e} {traceback_log}")
                failed_download(self)
                exit()
            '''elif check_if_online(dont_check=False, url="https://drive.google.com/"):
                            log("Couldnt connect to github, attempting to use google drive")
                            msg = QMessageBox()
                            msg.setWindowTitle(" ")
                            msg.setText(
                                f"Couldnt connect to GitHub! Attempting to download from Google Drive!\n(Please wait until the main window shows up, this will download in the backgroud.))"
                            )
                            msg.exec_()
                            GDrive.download_file_from_google_drive(
                                "1nOh01QQmet606W95ABBShrg5hFOuRwbo", f"{thisdir}/files/models.tar.gz"
                            )'''
        else:
            failed_download(self)
            exit()
    os.chdir(f"{thisdir}")


def choose_models(self):
    global window
    window = ChooseModels(self)
    window.show()


if check_for_individual_models() == None or check_for_each_binary() == False:

    class ChooseAI(QtWidgets.QMainWindow):
        def __init__(self):
            super(ChooseAI, self).__init__()
            self.ui = SelectAI.Ui_MainWindow()
            install_icons(self)
            self.ui.setupUi(self)
            self.pinFunctions()
            self.ui.GMFSSCUDACheckBox.setEnabled(isCUPY())
            self.show()

        def showDialogBox(self, message, displayInfoIcon=False):
            icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            if displayInfoIcon == True:
                msg.setIconPixmap(icon.pixmap(32, 32))
            msg.setText(f"{message}")

            msg.exec_()

        def pinFunctions(self):
            self.ui.modelsTabWidget.setTabEnabled(1, isCUDA())
            self.ui.InstallButton.clicked.connect(self.next)
            self.ui.RifeSettings.clicked.connect(lambda: choose_models(self))

        def next(self):
            global install_modules_dict
            install_modules_dict = returnModelList(self, settings)
            log(install_modules_dict)

            QApplication.closeAllWindows()

            return 0

    class ChooseModels(QtWidgets.QMainWindow):
        def __init__(self, parent):
            super(ChooseModels, self).__init__()
            self.ui = SelectModels.Ui_MainWindow()
            self.ui.setupUi(self)
            self.pinFunctions()

            self.show()
            self.main = parent

        def showDialogBox(self, message, displayInfoIcon=False):
            icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            if displayInfoIcon == True:
                msg.setIconPixmap(icon.pixmap(32, 32))
            msg.setText(f"{message}")

            msg.exec_()

        def pinFunctions(self):
            self.ui.next.hide()
            rife_pin_functions(self)
            try:
                with open(f"{thisdir}/models.txt", "r") as f:
                    for checkbox, option_name in rife_checkboxes(self):
                        if option_name in os.listdir(f"{thisdir}/models/rife/"):
                            checkbox.setChecked(True)

                    for i in f.readlines():
                        log(i)
                        i = i.replace("\n", "")
                        for checkbox, option_name in rife_checkboxes(self):
                            if option_name in i:
                                checkbox.setChecked(True)
            except:
                pass

        def checkbox_state_changed(self):
            rife_install_list = []

            for checkbox, option_name in rife_checkboxes(self):
                if checkbox.isChecked():
                    rife_install_list.append(option_name)
                self.main.ui.RifeCheckBox.setChecked(True)
            with open(f"{thisdir}/models.txt", "w") as f:
                for option in rife_install_list:
                    f.write(option + "\n")

    import src.programData.theme as theme

    app = QtWidgets.QApplication(sys.argv)
    theme.set_theme(app)

    window = ChooseAI()

    app.exec_()

    class Downloading(QtWidgets.QMainWindow):
        def __init__(self):
            super(Downloading, self).__init__()
            self.ui = DownloadUI.Ui_MainWindow()
            self.ui.setupUi(self)

            self.show()
            self.nextfunction()
            self.setWindowIcon(QIcon(f"{thisdir}/icons/logo v1.png"))

        def showDialogBox(self, message, displayInfoIcon=False):
            icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            if displayInfoIcon == True:
                msg.setIconPixmap(icon.pixmap(32, 32))
            msg.setText(f"{message}")

            msg.exec_()

        def nextfunction(self):
            try:
                logo = QIcon(f"{thisdir}/icons/logo v1.png")

                self.ui.logoPreview.setPixmap(logo.pixmap(256, 256))

                self.obj = Worker(self)
                self.thread = QThread()
                self.obj.intReady.connect(self.on_count_changed)
                self.obj.moveToThread(self.thread)
                self.obj.finished.connect(self.thread.quit)
                self.thread.started.connect(self.obj.install_modules)
                self.thread.start()
                self.obj.finished.connect(self.start_main)
            except Exception as e:
                traceback_info = traceback.format_exc()
                log(f"ERROR: {e} {traceback_info}")
                self.main.showDialogBox(e)

        def on_count_changed(self, list):
            downloaded_data = list[0]
            total_data = list[1]
            self.ui.downloadProgressBar.setMaximum(total_data)
            self.ui.downloadProgressBar.setValue(downloaded_data)
            downloaded_data_gb = str(downloaded_data / 1000000000)[:4]
            total_data_gb = str(total_data / 1000000000)[:4]

            self.ui.gbLabel.setText(f"{downloaded_data_gb}/{total_data_gb}GB")

        def start_main(self):
            for i in os.listdir(f"{thisdir}/files/"):
                if os.path.exists(f"{thisdir}/bin/") == False:
                    os.mkdir(f"{thisdir}/bin/")
                if i == "ffmpeg":
                    os.system(f'chmod +x "{thisdir}/files/ffmpeg"')
                    os.system(f'mv "{thisdir}/files/ffmpeg" "{thisdir}/bin/"')
                if i == "yt-dlp_linux":
                    os.system(f'chmod +x "{thisdir}/files/yt-dlp_linux"')
                    os.system(f'mv "{thisdir}/files/yt-dlp_linux" "{thisdir}/bin/"')
                if i == "glxinfo":
                    os.system(f'chmod +x "{thisdir}/files/glxinfo"')
                    os.system(f'mv "{thisdir}/files/glxinfo" "{thisdir}/bin/"')

                log(i)
                if ".zip" in i:
                    with ZipFile(f"{thisdir}/files/{i}", "r") as zip_ref:
                        name = i.replace(".zip", "")
                        if "-ncnn-vulkan" in name:
                            try:
                                original_ai_name_ncnn_vulkan = re.findall(
                                    r"[\w]*-ncnn-vulkan", name
                                )[0]
                                original_ai_name = original_ai_name_ncnn_vulkan.replace(
                                    "-ncnn-vulkan", ""
                                )
                                log(original_ai_name)
                            except:
                                pass
                        else:
                            original_ai_name = name
                            original_ai_name_ncnn_vulkan = name

                        zip_ref.extractall(f"{thisdir}/files/")

                    os.system(
                        f'mv "{thisdir}/files/{name}" "{settings.ModelDir}/{original_ai_name}"'
                    )
                    os.system(
                        f'chmod +x "{settings.ModelDir}/{original_ai_name}/{original_ai_name_ncnn_vulkan}"'
                    )
                    os.system(
                        f'chmod +x "{settings.ModelDir}/{original_ai_name}/upscayl-bin"'
                    )
                if ".tar.gz" in i:
                    with tarfile.open(f"{thisdir}/files/{i}", "r") as f:
                        f.extractall(f"{settings.ModelDir}/rife/")
                handleCUDAModels(i)
            clear_files()
            if check_for_individual_models != None:
                if check_if_online():
                    QApplication.closeAllWindows()

                    return 0
                else:
                    exit()
            else:
                failed_download(self)

                exit()

    if install_modules_dict:
        window = Downloading()
        app.exec_()
        app = None
        if os.path.isfile(f"{settings.ModelDir}/rife/rife-ncnn-vulkan") == True:
            QApplication.closeAllWindows()
        else:
            for file in os.listdir(f"{thisdir}/files"):
                if ".txt" not in file:
                    os.system(f'rm -rf "{thisdir}/files/{file}"')
            # this happens if program abruptly stops while downloading
    else:
        msg = QMessageBox()
        msg.setWindowTitle(" ")

        msg.setText(f"No models selected!")
        exit()


def excepthook(type, value, traceback):
    error_message = f"An unhandled exception occurred: {value}"
    log(f"ERROR: Unhandled exception! {traceback},{type},{error_message}")
    exit()

    QMessageBox.critical(None, "Error", error_message, QMessageBox.Ok)


sys.excepthook = excepthook

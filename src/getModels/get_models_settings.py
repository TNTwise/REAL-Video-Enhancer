import subprocess
from PyQt5.QtWidgets import QMainWindow
import src.getLinkVideo.get_vid_from_link as getLinkedVideo
from src.runAI.workers import *
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
)
from src.programData.return_data import *
import requests
import src.programData.thisdir
import src.getModels.SelectModels
from src.getModels.rifeModelsFunctions import *
from src.programData.checks import *
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
from zipfile import ZipFile
import tarfile
import src.misc.onProgramStart as programstart
import modules.Rife as rife
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox
from src.getModels.returnModelList import *

thisdir = src.programData.thisdir.thisdir()
rife_install_list = []


def handleCUDAModels(model: str = ""):
    if "rife" and "pkl" in model:
        os.system(
            f'mkdir -p "{thisdir}/models/rife-cuda/{model.replace(".","").replace("pkl","")}" '
        )
        os.system(
            f'cp "{thisdir}/files/{model}" "{thisdir}/models/rife-cuda/{model.replace(".","").replace("pkl","")}" '
        )
def deleteDownloaded():
        for i in os.listdir(f"{thisdir}/files/"):
            if os.path.isfile(i):
                if ".txt" not in i:
                    os.remove(f"{thisdir}/files/{i}")
        for i in os.listdir(f"{thisdir}/files/"):
            if ".txt" not in i:
                os.system(f'rm -rf "{thisdir}/files/{i}"')

class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(float)

    def __init__(self, parent):
        self.main = parent
        QThread.__init__(self, None)
    
    @pyqtSlot()
    def install_modules(self):
        
        deleteDownloaded()
        rife_install_list = []
       
        settings = Settings()

        os.system(f'touch "{thisdir}/models.txt"')
        with open(f"{thisdir}/models.txt", "r") as f:
            for i in f.readlines():
                i = i.replace("\n", "")
                rife_install_list.append(i)
                if "v4" in i:
                    rife_install_list.append(f"{i}-ensemble")
        if len(rife_install_list) == 0 and self.main.ui.RifeCheckBox.isChecked():
            rife_install_list.append("rife-v4.6")
        install_modules_dict = returnModelList(self.main, settings)
        log(install_modules_dict)
        total_size_in_bytes = 0
        data_downloaded = 0
        for link, name in install_modules_dict.items():
            response = requests.get(link, stream=True)
            total_size_in_bytes += int(response.headers.get("content-length", 0))
        if check_if_enough_space_for_install(total_size_in_bytes) == False:
            return 0
        for link, name in install_modules_dict.items():
            response = requests.get(link, stream=True)
            with open(f"{thisdir}/files/{name}", "wb") as f:
                for data in response.iter_content(1024):
                    f.write(data)
                    data_downloaded += 1024
                    self.intReady.emit(
                        data_downloaded / total_size_in_bytes
                    )  # sends back data to main thread

        for i in os.listdir(f"{thisdir}/files/"):
            if i == "rife-ncnn-vulkan":
                try:
                    os.mkdir(f"{settings.ModelDir}/rife/")
                except:
                    pass
                os.system(f'chmod +x "{thisdir}/files/rife-ncnn-vulkan"')

                os.system(
                    f'mv -f "{thisdir}/files/rife-ncnn-vulkan" "{settings.ModelDir}/rife/"'
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
                os.system(
                    f'chmod +x "{settings.ModelDir}/{original_ai_name}/upscayl-bin"'
                )

            if ".tar.gz" in i:
                with tarfile.open(f"{thisdir}/files/{i}", "r") as f:
                    f.extractall(f"{settings.ModelDir}/rife/")

            handleCUDAModels(i)

        try:
            for i in os.listdir(f"{settings.ModelDir}/rife/"):
                if (
                    i not in rife_install_list
                    and i != "rife-ncnn-vulkan"
                    and i in rife.default_models()
                ):
                    os.system(f'rm -rf "{settings.ModelDir}/rife/{i}"')
                    index = self.main.ui.defaultRifeModel.findText(i)

                    if index != -1:
                        self.main.ui.defaultRifeModel.removeItem(index)
                    else:
                        pass
        except:
            log("Rife not installed, but tried to remove anyway!")

        self.finished.emit()


class ChooseModels(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super(ChooseModels, self).__init__()
        self.ui = src.getModels.SelectModels.Ui_MainWindow()
        self.ui.setupUi(self)
        self.settings = Settings()
        self.ui.label_3.hide()
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

        for checkbox, option_name in rife_checkboxes(self):
            if option_name in os.listdir(f"{self.settings.ModelDir}/rife/"):
                checkbox.setChecked(True)
        for i in os.listdir(f"{self.settings.ModelDir}/rife/"):
            if (
                os.path.isfile(f"{self.settings.ModelDir}/rife/{i}") == False
                and i not in rife.default_models()
            ):
                checkbox = QCheckBox(i)
                checkbox.setChecked(True)  # Set the default state of the checkbox
                checkbox.setEnabled(False)
                self.ui.label_3.show()
                self.ui.custom_models.addWidget(checkbox)

    def checkbox_state_changed(self):
        rife_install_list = []

        for checkbox, option_name in rife_checkboxes(self):
            if checkbox.isChecked():
                rife_install_list.append(option_name)

        with open(f"{thisdir}/models.txt", "w") as f:
            for option in rife_install_list:
                f.write(option + "\n")


def remove_unchecked(self):
    if self.ui.RifeCheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/rife/"')

    if self.ui.RealESRGANCheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/realesrgan/"')

    if self.ui.Waifu2xCheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/waifu2x/"')

    if self.ui.CainCheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/ifrnet/"')

    if self.ui.RealCUGANCheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/realcugan/"')

    if self.ui.RealSRCheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/realsr/"')

    if self.ui.RifeCUDACheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/rife-cuda/"')

    if self.ui.RifeCUDACheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/realesrgan-cuda/"')

    if self.ui.RifeCUDACheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/custom_models_ncnn/"')

    if self.ui.SPANNCNNCheckBox.isChecked() == False:
        os.system(f'rm -rf "{self.settings.ModelDir}/span/"')


def run_install_models_from_settings(self):
    try:
        try:
            window.close()
        except:
            pass
        if model_warning(self):
            remove_unchecked(self)
            if check_if_online():
                self.setDisableEnable(True)
                self.thread5 = QThread()
                # Step 3: Create a worker object

                self.worker5 = Worker(self)

                # Step 4: Move worker to the thread
                self.worker5.moveToThread(self.thread5)
                # Step 5: Connect signals and slots
                self.thread5.started.connect(self.worker5.install_modules)
                self.worker5.finished.connect(self.thread5.quit)
                self.worker5.finished.connect(self.worker5.deleteLater)
                self.thread5.finished.connect(self.thread5.deleteLater)
                self.worker5.intReady.connect(displayProgressOnInstallBar)
                self.worker5.finished.connect(lambda: endDownload(self))
                global main
                main = self
                # Step 6: Start the thread

                self.thread5.start()

    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f"ERROR {e} {traceback_info}")
        return 0


def endDownload(self):
    deleteDownloaded()
    log(
        "==============================Finished model download============================"
    )
    self.setDisableEnable(False)
    # restart_app(self)
    programstart.onApplicationStart(self)
    self.ui.GeneralOptionsFrame.hide()
    self.ui.InstallModelsFrame.show()  # has to re-show frame as OnProgramStart defaults it to general
    if len(os.listdir(f"{thisdir}/files/")) < 1:
        self.ui.showDialogBox("Not enough space to install models!")
        log("ERROR: Not enough space")


def displayProgressOnInstallBar(downloaded):
    main.ui.installModelsProgressBar.setValue(int(downloaded * 100))


def get_rife(self):
    global window
    window = ChooseModels(self)
    window.show()

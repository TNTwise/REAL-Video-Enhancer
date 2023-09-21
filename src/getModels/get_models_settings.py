
import subprocess
from PyQt5.QtWidgets import  QMainWindow
import src.getLinkVideo.get_vid_from_link as getLinkedVideo
from src.workers import *
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
from src.return_data import *
import requests
import src.thisdir
import src.getModels.SelectModels
from src.checks import *
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot

thisdir = src.thisdir.thisdir()
rife_install_list = []

class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(list)
    def __init__(self,parent):
          self.main=parent
          QThread.__init__(self,None)
    @pyqtSlot()
    def install_modules(self):
            try:
                    os.system(f'touch "{thisdir}/models.txt"')
                    with open(f'{thisdir}/models.txt', 'r') as f:
                         for i in f.readlines():
                               rife_install_list.append(i)
                    
                    install_modules_dict={

'https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip':'realcugan-ncnn-vulkan-20220728-ubuntu.zip',
'https://github.com/nihui/cain-ncnn-vulkan/releases/download/20220728/cain-ncnn-vulkan-20220728-ubuntu.zip':'cain-ncnn-vulkan-20220728-ubuntu.zip',
'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan':'rife-ncnn-vulkan',

}
                    if self.main.ui.RifeCheckBox.isChecked() == True:
                          install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                    if self.main.ui.RealESRGANCheckBox.isChecked() == True:
                          install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/realesrgan-ncnn-vulkan-20220424-ubuntu.zip'] = 'realesrgan-ncnn-vulkan-20220424-ubuntu.zip'
                    if self.main.ui.Waifu2xCheckBox.isChecked() == True:
                          install_modules_dict['https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip'] = 'waifu2x-ncnn-vulkan-20220728-ubuntu.zip'
                    
                    for i in rife_install_list:
                                        install_modules_dict[f'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/{i}.tar.gz'] = f'{i}.tar.gz'
                    total_size_in_bytes=0
                    data_downloaded=0
                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        total_size_in_bytes+= int(response.headers.get('content-length', 0))

                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        with open(f'{thisdir}/files/{name}', 'wb') as f:
                                for data in response.iter_content(1024):
                                    f.write(data)
                                    data_downloaded+=1024
                                    #self.intReady.emit([int(data_downloaded),total_size_in_bytes]) # sends back data to main thread
            
            except Exception as e:
                self.main.showDialogBox(e)
class ChooseModels(QtWidgets.QMainWindow):
            def __init__(self,parent):
                super(ChooseModels, self).__init__()
                self.ui = src.getModels.SelectModels.Ui_MainWindow()
                self.ui.setupUi(self)
                self.pinFunctions()
                self.show()
                self.main = parent
            def showDialogBox(self,message,displayInfoIcon=False):
                icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
                msg = QMessageBox()
                msg.setWindowTitle(" ")
                if displayInfoIcon == True:
                    msg.setIconPixmap(icon.pixmap(32, 32))
                msg.setText(f"{message}")

                msg.exec_()
            def pinFunctions(self):
                checkboxes = [
                    (self.ui.rife, 'rife'),
                    (self.ui.rifeanime, 'rife-anime'),
                    (self.ui.rifehd, 'rife-HD'),
                    (self.ui.rifeuhd, 'rife-UHD'),
                    (self.ui.rife2, 'rife-v2'),
                    (self.ui.rife23, 'rife-v2.3'),
                    (self.ui.rife24, 'rife-v2.4'),
                    (self.ui.rife30, 'rife-v3.0'),
                    (self.ui.rife31, 'rife-v3.1'),
                    (self.ui.rife4, 'rife-v4'),
                    (self.ui.rife46, 'rife-v4.6')
                ]
                self.ui.next.hide()
                self.ui.rife.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rifeanime.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rifehd.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rifeuhd.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rife2.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rife23.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rife24.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rife30.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rife31.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rife4.stateChanged.connect(self.checkbox_state_changed)
                self.ui.rife46.stateChanged.connect(self.checkbox_state_changed)
                for checkbox,option_name in checkboxes:
                      
                    if option_name in os.listdir(f'{thisdir}/models/rife/'):
                          checkbox.setChecked(True)
                      
            def checkbox_state_changed(self):
                checkboxes = [
                    (self.ui.rife, 'rife'),
                    (self.ui.rifeanime, 'rife-anime'),
                    (self.ui.rifehd, 'rife-HD'),
                    (self.ui.rifeuhd, 'rife-UHD'),
                    (self.ui.rife2, 'rife-v2'),
                    (self.ui.rife23, 'rife-v2.3'),
                    (self.ui.rife24, 'rife-v2.4'),
                    (self.ui.rife30, 'rife-v3.0'),
                    (self.ui.rife31, 'rife-v3.1'),
                    (self.ui.rife4, 'rife-v4'),
                    (self.ui.rife46, 'rife-v4.6')
                ]
                rife_install_list = []

                

                for checkbox, option_name in checkboxes:
                    if checkbox.isChecked():
                        rife_install_list.append(option_name)

                with open(f'{thisdir}/models.txt', 'w') as f:
                    for option in rife_install_list:
                        f.write(option + '\n')

def run_install_models_from_settings(self):
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
    #self.worker5.progress.connect(self.reportProgress)
    #self.worker5.image_progress.connect(self.imageViewer)
    
    # Step 6: Start the thread
    
    self.thread5.start()
    
def get_rife(self):
    global window
    window = ChooseModels(self)
    window.show()
    



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
from zipfile import ZipFile
import tarfile
import src.onProgramStart as programstart
thisdir = src.thisdir.thisdir()
rife_install_list = []

class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(float)
    def __init__(self,parent):
          self.main=parent
          QThread.__init__(self,None)
    @pyqtSlot()
    
    def install_modules(self):
            
            for i in os.listdir(f'{thisdir}/files/'):
                         if '.txt' not in i:
                              try:
                                os.remove(f'{thisdir}/files/{i}')
                              except:
                                os.system(f'rm -rf "{thisdir}/files/{i}"')
            settings = Settings()
            try:
                    os.system(f'touch "{thisdir}/models.txt"')
                    with open(f'{thisdir}/models.txt', 'r') as f:
                         for i in f.readlines():
                               print(i)
                               i=i.replace('\n','')
                               rife_install_list.append(i)
                    
                    install_modules_dict={}

                    '''https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip':'realcugan-ncnn-vulkan-20220728-ubuntu.zip',
                    'https://github.com/nihui/cain-ncnn-vulkan/releases/download/20220728/cain-ncnn-vulkan-20220728-ubuntu.zip':'cain-ncnn-vulkan-20220728-ubuntu.zip',
                    '''
                    if self.main.ui.RifeCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/rife/') == False:
                          install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                    if self.main.ui.RifeCheckBox.isChecked() == False:
                         os.system(f'rm -rf "{settings.ModelDir}/rife/"')

                    if self.main.ui.RealESRGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realesrgan') == False:
                          install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/realesrgan-ncnn-vulkan-20220424-ubuntu.zip'] = 'realesrgan-ncnn-vulkan-20220424-ubuntu.zip'
                    if self.main.ui.RealESRGANCheckBox.isChecked() == False:
                         os.system(f'rm -rf "{settings.ModelDir}/realesrgan/"')
                    
                    if self.main.ui.Waifu2xCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/waifu2x') == False:
                          install_modules_dict['https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip'] = 'waifu2x-ncnn-vulkan-20220728-ubuntu.zip'
                    if self.main.ui.Waifu2xCheckBox.isChecked() == False:
                         os.system(f'rm -rf "{settings.ModelDir}/waifu2x/"')
                    
                    if self.main.ui.CainCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/ifrnet') == False:
                          install_modules_dict['https://github.com/nihui/ifrnet-ncnn-vulkan/releases/download/20220720/ifrnet-ncnn-vulkan-20220720-ubuntu.zip'] = 'ifrnet-ncnn-vulkan-20220720-ubuntu.zip'
                    if self.main.ui.CainCheckBox.isChecked() == False:
                         os.system(f'rm -rf "{settings.ModelDir}/ifrnet/"')
                    
                    if self.main.ui.RealCUGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realcugan') == False:
                          install_modules_dict['https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip'] = 'realcugan-ncnn-vulkan-20220728-ubuntu.zip'
                    if self.main.ui.RealCUGANCheckBox.isChecked() == False:
                         os.system(f'rm -rf "{settings.ModelDir}/realcugan/"')

                    for i in rife_install_list:
                            if os.path.exists(f'{settings.ModelDir}/rife/rife-ncnn-vulkan') == False:
                                 install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                            if os.path.exists(f'{settings.ModelDir}/rife/{i}') == False:
                                    install_modules_dict[f'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/{i}.tar.gz'] = f'{i}.tar.gz'
                    if rife_install_list == [] and self.main.ui.RifeCheckBox.isChecked():
                         install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                         rife_install_list.append('rife-v4.6')
                    total_size_in_bytes=0
                    data_downloaded=0
                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        total_size_in_bytes+= int(response.headers.get('content-length', 0))
                    if check_if_enough_space_for_install(total_size_in_bytes) == False:
                         return 0
                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        with open(f'{thisdir}/files/{name}', 'wb') as f:
                                for data in response.iter_content(1024):
                                    f.write(data)
                                    data_downloaded+=1024
                                    self.intReady.emit(data_downloaded/total_size_in_bytes)# sends back data to main thread
                    os.system(f'mv "{thisdir}/files/rife-ncnn-vulkan" "{settings.ModelDir}/rife/"')
                    os.system(f'mv "{thisdir}/files/rife-ncnn-vulkan" "{settings.ModelDir}/rife/"')
                    for i in os.listdir(f'{thisdir}/files/'):
                        
                             
                        if '.zip' in i:

                            with ZipFile(f'{thisdir}/files/{i}', 'r') as zip_ref:
                                name=i.replace('.zip','')
                                original_ai_name_ncnn_vulkan = re.findall(r'[\w]*-ncnn-vulkan', name)[0]
                                original_ai_name = original_ai_name_ncnn_vulkan.replace('-ncnn-vulkan','')


                                zip_ref.extractall(f'{thisdir}/files/')

                            os.system(f'mv "{thisdir}/files/{name}" "{settings.ModelDir}/{original_ai_name}"')
                            os.system(f'chmod +x "{settings.ModelDir}/{original_ai_name}/{original_ai_name_ncnn_vulkan}"')

                        if '.tar.gz' in i:
                            with tarfile.open(f'{thisdir}/files/{i}','r') as f:
                                f.extractall(f'{settings.ModelDir}/rife/')


                    os.system(f'mv "{thisdir}/files/rife-ncnn-vulkan" "{settings.ModelDir}/rife/"')
                    os.system(f'chmod +x "{settings.ModelDir}/rife/rife-ncnn-vulkan"')
                    for i in os.listdir(f'{thisdir}/files/'):
                         if '.txt' not in i:
                              os.remove(f'{thisdir}/files/{i}')
                    for i in os.listdir(f'{settings.ModelDir}/rife/'):
                        if i not in rife_install_list and i != 'rife-ncnn-vulkan':
                             os.system(f'rm -rf "{settings.ModelDir}/rife/{i}"')
                    self.finished.emit()
            except Exception as e:
                self.main.showDialogBox(e)
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
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
    try:
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
    except:
        
        return 0
def endDownload(self):
     
     self.setDisableEnable(False)
     #restart_app(self)
     programstart.onApplicationStart(self)
     self.ui.GeneralOptionsFrame.hide()
     self.ui.InstallModelsFrame.show() # has to re-show frame as OnProgramStart defaults it to general
     if len(os.listdir(f'{thisdir}/files/')) < 1:
          self.ui.showDialogBox('Not enough space to install models!')
def displayProgressOnInstallBar(downloaded):
    main.ui.installModelsProgressBar.setValue(int(downloaded*100))
    
def get_rife(self):
    global window
    window = ChooseModels(self)
    window.show()
    


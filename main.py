from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
import mainwindow
import os
from threading import *
from src.settings import *
from src.return_data import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.start as start

import src.get_models as get_models
from time import sleep

thisdir = os.getcwd()
homedir = os.path.expanduser(r"~")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        
        #Define Variables
        self.input_file = ''
        self.output_folder = ''
        self.output_folder = settings.OutputDir 

        self.pin_functions()
        self.show()

    def pin_functions(self):

        self.ui.Input_video_rife.clicked.connect(self.openFileNameDialog)
        self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)
        
        self.ui.RifeStart.clicked.connect(self.startRife)

        # list every model downloaded, and add them to the list
       
        model_filepaths = ([x[0] for x in os.walk(f'{thisdir}/rife-vulkan-models/')])
        models = []
        for model_filepath in model_filepaths:
            if 'rife' in os.path.basename(model_filepath):
                models.append(os.path.basename(model_filepath))
        
        
        for model in models:

            
            model = model.replace('r',"R")
            model = model.replace('v','V')
            model = model.replace('a','A')
            self.ui.Rife_Model.addItem(f'{model}')#Adds model to GUI.
            if model == 'Rife-V2.3':
                self.ui.Rife_Model.setCurrentText(f'{model}')
            
    def openFileNameDialog(self):

        self.input_file = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]

    def openFolderDialog(self):
        
        self.output_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
    

    def updateRifeProgressBar(self,times,start_value):
        videoName = VideoName.return_video_name(f'{self.input_file}')
        while ManageFiles.isfolder(f'{settings.RenderDir}/{videoName}/output_frames/') == False:
            sleep(1)
        

        total_input_files = len(os.listdir(f'{settings.RenderDir}/{videoName}/input_frames/'))
        total_output_files = total_input_files * times
        self.ui.RifePB.setMaximum(total_output_files)
        print(total_output_files)
        print(videoName)
        sleep(1)
        while ManageFiles.isfolder(f'{settings.RenderDir}/{videoName}/') == True:
                
                
                files_processed = len(os.listdir(f'{settings.RenderDir}/{videoName}/output_frames/'))
                
                self.ui.RifePB.setValue(files_processed)
                sleep(1)
        self.ui.RifePB.setValue(total_output_files)
        return 0

    def _setStyle(self,color):
        self.ui.RifeStart.setStyleSheet(f'color: {color};')
        self.ui.Input_video_rife.setStyleSheet(f'color: {color};')
        self.ui.Output_folder_rife.setStyleSheet(f'color: {color};')
        self.ui.Rife_Model.setStyleSheet(f"QComboBox {{ color: {color}; }}")
        self.ui.Rife_Times.setStyleSheet(f"QComboBox {{ color: {color}; }}")

    def setDisableEnable(self,mode):
        self.ui.RifeStart.setDisabled(mode)
        self.ui.Input_video_rife.setDisabled(mode) 
        self.ui.Output_folder_rife.setDisabled(mode)
        self.ui.Rife_Model.setDisabled(mode)
        self.ui.Rife_Times.setDisabled(mode)

    def endRife(self):
        self.rifeThread.join()
        
        current_palette = app.style().standardPalette()
        self.setDisableEnable(False)
        if current_palette.color(current_palette.WindowText).lightness() > 127:
            self._setStyle('white')
        else:
            self._setStyle('black')

    def startRife(self): #should prob make this different, too similar to start_rife but i will  think of something later prob

        if self.input_file != '':
            self.setDisableEnable(True)
            self._setStyle('gray')
            
           
            if int(self.ui.Rife_Times.currentText()[0]) == 2:
                self.rifeThread = Thread(target=lambda: self.start_rife((self.ui.Rife_Model.currentText().lower()),2,self.input_file,self.output_folder,1))
            if int(self.ui.Rife_Times.currentText()[0]) == 4:
                self.rifeThread = Thread(target=lambda: self.start_rife((self.ui.Rife_Model.currentText().lower()),4,self.input_file,self.output_folder,2))
            if int(self.ui.Rife_Times.currentText()[0]) == 8:
                self.rifeThread = Thread(target=lambda: self.start_rife((self.ui.Rife_Model.currentText().lower()),4,self.input_file,self.output_folder,3))
            self.rifeThread.start()
                
            Thread(target=self.endRife).start()
        else:
            self.showDialogBox("No input file selected.")

    
    def start_rife(self,model,times,videopath,outputpath,end_iteration,renderdir=thisdir):
        
        videoName = VideoName.return_video_name(fr'{videopath}')
        
        start.start(renderdir,videoName,videopath)
        
                #change progressbar value
    
        for i in range(end_iteration):
            if times == 2:
                Thread(target=lambda: self.updateRifeProgressBar(2,0)).start()
            os.system(f'"{thisdir}/rife-vulkan-models/rife-ncnn-vulkan" -m  {model} -i {renderdir}/{videoName}_temp/input_frames/ -o {renderdir}/{videoName}_temp/output_frames/')
        
        
        start.end(renderdir,videoName,videopath,times,outputpath)

    def showDialogBox(self,message):
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        msg.setText(f"{message}")
        msg.exec_()
    


settings = Settings()


    
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
sys.exit(app.exec_())
    


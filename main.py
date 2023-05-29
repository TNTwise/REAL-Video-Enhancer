#!/usr/bin/python3

from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox, QListWidget, QListWidgetItem
from PyQt5.QtGui import QTextCursor
import mainwindow
import os
from threading import *
from src.settings import *
from src.return_data import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.start as start
import src.workers as workers
import time
#import src.get_models as get_models
from time import sleep
import src.get_models as get_models
thisdir = os.getcwd()
homedir = os.path.expanduser(r"~")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.SettingsMenus.clicked.connect(self.settings_menu)
        
        self.def_var()
        self.pin_functions()
        self.show()
    def calculateETA(self):
        videoName = VideoName.return_video_name(f'{self.input_file}')
        self.ETA=None
        while os.path.exists(f'{self.render_folder}/{videoName}_temp/input_frames/'):
            
            total_iterations = len(os.listdir(f'{self.render_folder}/{videoName}_temp/input_frames/')) * self.times
        
            start_time = time.time()
            
            sleep(1)
            for i in range(total_iterations):
                # Do some work for each iteration
                
                try:
                    
                        
                        
                    completed_iterations = len(os.listdir(f'{self.render_folder}/{videoName}_temp/output_frames/'))
                    
                    # Increment the completed iterations counter
                    sleep(1)

                    # Estimate the remaining time
                    elapsed_time = time.time() - start_time
                    time_per_iteration = elapsed_time / completed_iterations
                    remaining_iterations = total_iterations - completed_iterations
                    remaining_time = remaining_iterations * time_per_iteration
                    remaining_time = int(remaining_time) 
                    # Print the estimated time remaining
                    #convert to hours, minutes, and seconds
                    hours = remaining_time // 3600
                    remaining_time-= 3600*hours
                    minutes = remaining_time // 60
                    remaining_time -= minutes * 60
                    seconds = remaining_time
                    if minutes < 10:
                        minutes = str(f'0{minutes}')
                    if seconds < 10:
                        seconds = str(f'0{seconds}')
                    self.ETA = f'ETA: {hours}:{minutes}:{seconds}'
                except:
                    self.ETA = None
    def reportProgress(self, n):
        
        videoName = VideoName.return_video_name(f'{self.input_file}')
        # fc is the total file count after interpolation
        fc = int(VideoName.return_video_frame_count(f'{self.input_file}') * self.times)
        if self.i==1:
            self.addLinetoLogs(f'Starting {self.times}X Render')
            self.original_fc=fc/self.times # this makes the original file count. which is the file count before interpolation
            self.i=2
        if self.times == 4:
            fc += (fc/2) #This line adds in to the total file count the previous 2x interpolation for total file count
        if self.times == 8:
            fc += (fc)
            fc += (fc/2)
        
        if self.addLast == True: #this checks for addLast, which is set after first interpolation in 4X, and if its true it will add the original file count * 2 onto that
            n+=self.original_fc*2
            
        n=int(n)
        fc = int(fc)
        self.ui.RifePB.setValue(n*int(self.times/2))
        self.ui.processedPreview.setText(f'Files Processed: {n} / {fc}')
        
        
        

        if self.ETA != None:
            self.ui.ETAPreview.setText(self.ETA)
        if self.i == 1 and os.path.exists(f'{self.render_folder}/{videoName}_temp/output_frames'):
            self.ui.logsPreview.append(f'Starting {self.times}X Render')
            self.i = 2
    def runPB(self,videoName,times):
        self.addLast=False
        self.i=1
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
       
        self.worker = workers.pb2X(self,videoName)
        
        self.times = times
        Thread(target=self.calculateETA).start()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        self.worker.finished.connect(
            self.endRife
        )
        self.worker.finished.connect(
            lambda: self.ui.RifePB.setValue(self.ui.RifePB.maximum())
        )
        self.worker.finished.connect(
            lambda: self.ui.ETAPreview.setText('ETA: 00:00:00')
        )
        
    
        
    def def_var(self):
        #Define Variables
        self.input_file = ''
        self.output_folder = ''
        self.output_folder = settings.OutputDir 
        self.videoQuality = settings.videoQuality
        self.encoder = settings.Encoder
        if os.path.exists(f"{settings.RenderDir}") == False:
            settings.change_setting('RenderDir',f'{thisdir}')
        self.render_folder = settings.RenderDir
    def settings_menu(self):
        item = self.ui.SettingsMenus.currentItem()
        if item.text() == "Video Options":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.show()
        if item.text() == "Render Options":
            self.ui.RenderOptionsFrame.show()
            self.ui.VideoOptionsFrame.hide()
    
    def pin_functions(self):
        if self.encoder == '264':
            self.ui.EncoderCombo.setCurrentIndex(0)
        if self.encoder == '265':
            self.ui.EncoderCombo.setCurrentIndex(1)
        if self.videoQuality == '10':
            self.ui.VidQualityCombo.setCurrentText('Lossless')
        if self.videoQuality == '14':
            self.ui.VidQualityCombo.setCurrentText('High')
        if self.videoQuality == '18':
            self.ui.VidQualityCombo.setCurrentText('Medium')
        if self.videoQuality == '22':
            self.ui.VidQualityCombo.setCurrentText('Low')
        self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")
        self.ui.RenderDirButton.clicked.connect(self.selRenderDir)
        self.ui.verticalTabWidget.setCurrentWidget(self.ui.verticalTabWidget.findChild(QWidget, 'Rife'))
        self.ui.Input_video_rife.clicked.connect(self.openFileNameDialog)
        self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)
        self.ui.VideoOptionsFrame.hide()
        self.ui.RenderOptionsFrame.hide()
        self.ui.RifeStart.clicked.connect(self.startRife)

        self.ui.EncoderCombo.currentIndexChanged.connect(self.selEncoder)
        #apparently adding multiple currentindexchanged causes a memory leak unless i sleep, idk why it does this but im kinda dumb
        sleep(0.01)
        self.ui.VidQualityCombo.currentIndexChanged.connect(self.selVidQuality)

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
    
    def selRenderDir(self):

        self.render_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
        settings.change_setting("RenderDir",f"{self.render_folder}")
        
        self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")

    def selEncoder(self):
        if '.264' in self.ui.EncoderCombo.currentText():
            
            settings.change_setting('Encoder','264')
        if '.265' in self.ui.EncoderCombo.currentText():
            settings.change_setting('Encoder','265')
        self.encoder = settings.Encoder
    
    def selVidQuality(self):
        if self.ui.VidQualityCombo.currentText() == 'Lossless':
            settings.change_setting('videoQuality', '10')
        if self.ui.VidQualityCombo.currentText() == 'High':
            settings.change_setting('videoQuality', '14')
        if self.ui.VidQualityCombo.currentText() == 'Medium':
            settings.change_setting('videoQuality', '18')
        if self.ui.VidQualityCombo.currentText() == 'Low':
            settings.change_setting('videoQuality', '22')
        self.videoQuality = settings.videoQuality
        
    def openFileNameDialog(self):

        self.input_file = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]

    def openFolderDialog(self):
        
        self.output_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
    

   

    
    def setDisableEnable(self,mode):
        self.ui.RifeStart.setDisabled(mode)
        self.ui.Input_video_rife.setDisabled(mode) 
        self.ui.Output_folder_rife.setDisabled(mode)
        self.ui.Rife_Model.setDisabled(mode)
        self.ui.Rife_Times.setDisabled(mode)

    def endRife(self):
        
        self.setDisableEnable(False)
        self.show()
    #The code below here is a multithreaded mess, i will fix later with proper pyqt implementation
    def startRife(self): #should prob make this different, too similar to start_rife but i will  think of something later prob

        if self.input_file != '':
            
            self.setDisableEnable(True)
            
            
            self.ui.logsPreview.append(f'Extracting Frames')

            if int(self.ui.Rife_Times.currentText()[0]) == 2:
                self.rifeThread = Thread(target=lambda: self.start_rife((self.ui.Rife_Model.currentText().lower()),2,self.input_file,self.output_folder,1))
            if int(self.ui.Rife_Times.currentText()[0]) == 4:
                self.rifeThread = Thread(target=lambda: self.start_rife((self.ui.Rife_Model.currentText().lower()),4,self.input_file,self.output_folder,2))
            if int(self.ui.Rife_Times.currentText()[0]) == 8:
                self.rifeThread = Thread(target=lambda: self.start_rife((self.ui.Rife_Model.currentText().lower()),8,self.input_file,self.output_folder,3))
            self.rifeThread.start()
                
        else:
            self.showDialogBox("No input file selected.")

    
    def start_rife(self,model,times,videopath,outputpath,end_iteration):
        
        videoName = VideoName.return_video_name(fr'{videopath}')
        self.ui.ETAPreview.setText('ETA:')
        self.ui.processedPreview.setText('Files Processed:')
        # Have to put this before otherwise it will error out ???? idk im not good at using qt.....
                
                
        #self.runLogs(videoName,times)
       
        start.start(self.render_folder,videoName,videopath)
        total_input_files = len(os.listdir(f'{settings.RenderDir}/{videoName}_temp/input_frames/'))
        total_output_files = total_input_files * times 
        if times == 4:
            total_output_files += (total_output_files*2)
        if times == 8:
            total_output_files += (total_output_files*4)
            total_output_files += (total_output_files*2)
        self.runPB(videoName,times)
        
        self.ui.RifePB.setMaximum(total_output_files)
                #change progressbar value
    
        for i in range(end_iteration):
            if i != 0:
                if times == 4: 
                    self.addLast=True
                    self.ui.RifePB.setValue(int(len(os.listdir(f'{self.render_folder}/{videoName}_temp/output_frames/'))))
                os.system(fr'rm -rf "{self.render_folder}/{videoName}_temp/input_frames/"  &&  mv "{self.render_folder}/{videoName}_temp/output_frames/" "{self.render_folder}/{videoName}_temp/input_frames" && mkdir -p "{self.render_folder}/{videoName}_temp/output_frames"')
                
                
            os.system(f'"{thisdir}/rife-vulkan-models/rife-ncnn-vulkan" -m  {model} -i {self.render_folder}/{videoName}_temp/input_frames/ -o {self.render_folder}/{videoName}_temp/output_frames/')
        
        if os.path.exists(f'{self.render_folder}/{videoName}_temp/output_frames/') == False or os.path.isfile(f'{self.render_folder}/{videoName}_temp/audio.m4a') == False:
            self.showDialogBox('Output frames or Audio file does not exist. Did you accidently delete them?')
        else:
            start.end(self.render_folder,videoName,videopath,times,outputpath, self.videoQuality,self.encoder)
            
    def showDialogBox(self,message):
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        msg.setText(f"{message}")
        msg.exec_()
    
    
    def addLinetoLogs(self,line):
        
        self.ui.logsPreview.append(f'{line}')
    def removeLastLineInLogs(self):
        
        cursor = self.ui.logsPreview.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Move the cursor to the beginning of the last line
        cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.MoveAnchor)
        cursor.movePosition(QTextCursor.PreviousBlock, QTextCursor.KeepAnchor)

        # Remove the selected text (the last line)
        cursor.removeSelectedText()
        
        self.ui.logsPreview.setTextCursor(cursor)
        
if os.path.isfile(f'{thisdir}/files/settings.txt') == False:
    ManageFiles.create_folder(f'{thisdir}/files')
    ManageFiles.create_file(f'{thisdir}/files/settings.txt')
settings = Settings()


    
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
sys.exit(app.exec_())
    


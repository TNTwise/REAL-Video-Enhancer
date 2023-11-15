#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
from src.settings import *
from threading import Thread
import src.runAI.transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
import os
from modules.commands import *
import src.thisdir
import modules.interpolate as interpolate
import src.onProgramStart as onProgramStart
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
def bubble_sort(arr):
  iteration_count = 0
  list_at_end=[]
  for i in range(len(arr)):
    for idx in range(len(arr) - i - 1):
      var1 = arr[idx].replace('rife-v','') 
      var1=var1.replace('.','')
      var2 = arr[idx+1].replace('rife-v','') 
      var2=var2.replace('.','')
      iteration_count += 1
      try:
        if int(var1) > int(var2):
            arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
      except:
          list_at_end.append(arr[idx])
  for i in list_at_end:
      if i not in arr:
        
            arr.append(i)

def modelOptions(self):
    self.times=2
    self.render='rife'
    self.ui.Rife_Model.clear()
    self.ui.Rife_Times.clear()
    self.ui.FPSPreview.setText('FPS:')
    
    self.ui.Rife_Times.addItem('2X')
    self.ui.Rife_Times.addItem('4X')
    self.ui.Rife_Times.addItem('8X')
    self.ui.Rife_Times.currentIndexChanged.connect(self.showChangeInFPS)
    try:
        self.ui.Rife_Model.currentIndexChanged.disconnect()
    except:
        pass
    self.ui.Rife_Times.setCurrentIndex(0)
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()
    self.showChangeInFPS()
    try:
        self.ui.RifeStart.clicked.disconnect() 
    except:
        pass
    self.ui.RifeStart.clicked.connect(lambda: interpolate.start_interpolation(self,'rife-ncnn-vulkan'))
    models = self.get_models_from_dir('rife')
    bubble_sort(models)
    print(models)
    
    if len (self.get_models_from_dir("rife")) > 0:
        self.ui.Rife_Model.addItems(models)
        model_list=[]
        for i in range(self.ui.defaultRifeModel.count()):
            item_text = self.ui.defaultRifeModel.itemText(i)
            model_list.append(item_text)
        for i in models:
            if i not in model_list:
            
                print(f'added model {i}')
                self.ui.defaultRifeModel.addItem(i)
        if  f'{self.settings.DefaultRifeModel}' in models:
                self.ui.Rife_Model.setCurrentText(f'{self.settings.DefaultRifeModel}')
        else:
            self.settings.change_setting(f'DefaultRifeModel',f'{self.get_models_from_dir("rife")[-1]}')
    
            self.ui.Rife_Model.setCurrentText(f'{self.settings.DefaultRifeModel}')
    
    if 'v4' in self.ui.Rife_Model.currentText():

        self.ui.Rife_Times.setEnabled(True)
    else:
        self.ui.Rife_Times.setEnabled(False)


def default_models():
    return ['rife-v2.3','rife-v4.6','rife-v4.8','rife','rife-anime','rife-HD','rife-UHD','rife-v2','rife-v2.4','rife-v3.0','rife-v3.1','rife-v4','rife-v4.1','rife-v4.2','rife-v4.3','rife-v4.4','rife-v4.5','rife-v4.9','rife-v4.10','rife-v4.11','rife-v4.12','rife-v4.7']
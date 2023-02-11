import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import os
from glob import glob
import re
thisdir = os.getcwd()

class getModels:
    def __init__():
        super().__init__()



class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        # Sets window title
        self.setWindowTitle("Rife - ESRGAN - App - Linux")
        
        self.setLayout(qtw.QVBoxLayout())
        self.layout_rife()
        #Show the app
        self.show()

    def layout_rife(self):
        my_label = qtw.QLabel("Pick rife ver")
        self.layout().addWidget(my_label)
        
        # Change font size

        my_label.setFont(qtg.QFont("Helvetica", 18))
        my_combo = qtw.QComboBox(self)
        os.chdir(f"{thisdir}/rife-vulkan-models/")
        rife_models = glob("rife-*")
        os.chdir(f"{thisdir}")
        my_combo.addItem('rife')
        for i in rife_models:
            if i != "rife-ncnn-vulkan":
                my_combo.addItem(i)
        
        self.layout().addWidget(my_combo)
    

#start the app

app = qtw.QApplication([])
mw = MainWindow()

app.exec_()
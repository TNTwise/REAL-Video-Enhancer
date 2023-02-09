import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg

class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        #Add a title
        self.setWindowTitle("Rife - ESRGAN - APP - Linux")
        

        # Set layout

        self.setLayout(qtw.QVBoxLayout())
        
        # Create a label
        my_label = qtw.QLabel("Pick rife ver")
        self.layout().addWidget(my_label)
        
        # Change font size

        my_label.setFont(qtg.QFont("Helvetica", 18))

        def entry(): # Create an entry box 
            my_entry = qtw.QLineEdit()
            my_entry.setObjectName("name_field")
            my_entry.setText("")
            self.layout().addWidget(my_entry)
        
        
        my_combo = qtw.QComboBox(self)
            #ADD items
        my_combo.addItem("2.3")
        my_combo.addItem("2.4")
        self.layout().addWidget(my_combo)
        
        my_button = qtw.QPushButton("Press me", clicked=lambda: press_it())
        self.layout().addWidget(my_button)
        def press_it():
            my_label.setText(f'You picked {my_combo.currentText()}')
        #Show the app
        self.show()

        
        
app = qtw.QApplication([])
mw = MainWindow()

app.exec_()
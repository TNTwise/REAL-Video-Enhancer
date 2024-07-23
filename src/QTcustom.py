from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QThread,QObject

import requests
styleSheet = (u"QLabel{\n"  
"	color: #fff;\n"
"}\n"
"QLineEdit{\n"
"color: #fff;\n"
"}\n"
"#centralwidget{\n"
"	background-color:#1f232a;\n"
"}\n"
"#leftMenuSubContainer{\n"
"	background-color:#16191d;\n"
"   border-radius: 30px;\n"
"}\n"
"#bottomMenuSubContainer{\n"
"	background-color:#16191d;\n"
"   border-radius: 30px;\n"
"}\n"
"\n"
"QProgressBar{\n"
"    background-color:#2c313c;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 10px;\n"
"\n"
"}\n"
"QProgressDialog{\n"
"	background-color:#16191d;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 25px;\n"
"}\n"
"QPushButton{\n"
"    background-color:#2c313c;\n"
"	text-align:left;\n"
"	padding:5px 10px;\n"
"	border-radius: 10px;\n"
"    color: #fff;\n"
"}\n"
"QPushButton:checked{\n"
"	background-color:#676e7b;\n"
"}\n"
"QPushButton:hover{\n"
"	background-color:#343b47;\n"
"}\n"
"\n"
"")

class DownloadAndReportToQTThread(QThread):
    """
    Downloads a file while reporting the actual bytes downloaded
    """
    finished = QtCore.Signal()
    progress = QtCore.Signal(int)
    def __init__(self, parent=None):
        QThread.__init__(self, parent)
    def run(self,link,downloadLocation):
        
        response = requests.get(
        link,
        stream=True,
        )
        totalSize = 0
        with open(downloadLocation, "wb") as f:
            chunk_size = 128
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                totalSize+=chunk_size
                self.progress.emit(totalSize)
        self.finished.emit()

class DownloadProgressPopup(QtWidgets.QProgressDialog):
    """
    Runs a download of a file in another thread while reporitng progress to a qt progressbar popup
    """
    def __init__(self, link: str, downloadLocation: str):
        super().__init__()
        self.link = link
        self.downloadLocation = downloadLocation
        self.workerThread = QThread()
        self.worker = DownloadAndReportToQTThread()
        totalSize = int(requests.get(link).headers['Content-Length'])
        self.setStyleSheet(styleSheet)
        self.setRange(0,totalSize)
        self.startDownload()
        self.exec()
    """
    Initializes all threading bs
    """
    def startDownload(self):
        self.worker.moveToThread(self.workerThread)
        self.workerThread.started.connect(lambda: self.worker.run(link=self.link, downloadLocation=self.downloadLocation))
        self.worker.progress.connect(self.setProgress)
        self.worker.finished.connect(self.workerThread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.close)
        self.workerThread.start()

    def setProgress(self, value):
        
        self.setValue(value)

        

        
            

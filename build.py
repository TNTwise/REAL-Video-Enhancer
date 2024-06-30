import os
import PyQt5.uic as uic
def buildenv():
    
    filelist = [
    "mainwindow",
    os.path.join("src","getModels","SelectModels"),
    os.path.join("src","getModels","Download"),
    os.path.join("src","getModels","SelectAI"),
    os.path.join("src","getLinkVideo","get_vid_from_link")
    ]
    for file in filelist:
        with open(f"{file}.py", 'w') as f:
            uic.compileUi(f"{file}.ui",f)
    
def buidNCNNLinux():
    pass
def buildCUDALinux():
    pass
def buildROCmLinux():
    pass
def buildNCNNFlatpakLinux():
    pass
def buildNCNNMacOS():
    pass

buildenv()
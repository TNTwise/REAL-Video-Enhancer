all: BuildApp
BuildApp:
        
		python3 -m PyInstaller --icon=icons/logov1.png  --hidden-import='PyQt5.QFileDialog' --noconfirm main.py && cp -r bin/ dist/main/ && cp -r icons/ dist/main/

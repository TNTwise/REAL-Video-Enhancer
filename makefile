all: BuildApp
BuildApp:
        
		python3 -m PyInstaller -F --icon=icons/logov1.png  --hidden-import='PyQt5.QFileDialog' --noconfirm main.py && cp -r bin/ dist/ && cp -r icons/ dist/

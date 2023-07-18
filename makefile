all: BuildApp
BuildApp:
        
		python3 -m PyInstaller  --hidden-import='PyQt5.QFileDialog'  --add-data="bin/ffmpeg":"./bin/" --add-data="icons/*":"./icons" --noconfirm  main.py

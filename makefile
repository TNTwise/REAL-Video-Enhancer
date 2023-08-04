all: BuildApp
BuildApp:
        	python3 -m PyQt5.uic.pyuic mainwindow.ui > mainwindow.py
		python3 -m PyQt5.uic.pyuic src/getModels/SelectModels.ui > src/getModels/SelectModels.py
		python3 -m PyQt5.uic.pyuic src/getModels/Download.ui > src/getModels/Download.py
		python3 -m PyInstaller -F  --hidden-import='PyQt5.QFileDialog'  --add-data="bin/ffmpeg":"./bin/" --add-data="icons/*":"./icons" --noconfirm  main.py

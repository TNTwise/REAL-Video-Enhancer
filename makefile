all: BuildApp
BuildApp:
		python3 -m PyQt5.uic.pyuic mainwindow.ui > mainwindow.py
		rm -rf dist
		rm -rf build
		python3 -m PyQt5.uic.pyuic src/getModels/SelectModels.ui > src/getModels/SelectModels.py
		python3 -m PyQt5.uic.pyuic src/getModels/Download.ui > src/getModels/Download.py
		python3 -m PyInstaller --noconfirm  main.py

all: BuildApp
BuildApp:
		python3 -m PyQt5.uic.pyuic mainwindow.ui > mainwindow.py
		python3 -m PyQt5.uic.pyuic src/getModels/SelectModels.ui > src/getModels/SelectModels.py
		python3 -m PyQt5.uic.pyuic src/getModels/Download.ui > src/getModels/Download.py
		cxfreeze -c main.py --target-dir dist/

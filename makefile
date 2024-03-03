all: BuildApp
BuildApp:
		python3 -m PyQt5.uic.pyuic mainwindow.ui > mainwindow.py
		rm -rf dist
		rm -rf build
		mkdir dist
		python3 -m PyQt5.uic.pyuic mainwindow.ui > mainwindow.py
		python3 -m PyQt5.uic.pyuic src/getModels/SelectModels.ui > src/getModels/SelectModels.py
		python3 -m PyQt5.uic.pyuic src/getModels/Download.ui > src/getModels/Download.py
		python3 -m PyQt5.uic.pyuic src/getModels/SelectAI.ui > src/getModels/SelectAI.py
		python3 -m PyQt5.uic.pyuic src/getLinkVideo/get_vid_from_link.ui > src/getLinkVideo/get_vid_from_link.py
		python3 -m PyInstaller main.py

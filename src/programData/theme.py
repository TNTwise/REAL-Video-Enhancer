from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from src.programData.settings import Settings


def set_theme(app):
    settings = Settings()
    if settings.Theme == "Dark":
        app.setStyle("Fusion")
        
        # Now use a palette to switch to dark colors:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.black)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Disabled, QPalette.Base, QColor(49, 49, 49))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(90, 90, 90))
        palette.setColor(QPalette.Disabled, QPalette.Button, QColor(42, 42, 42))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(90, 90, 90))
        palette.setColor(QPalette.Disabled, QPalette.Window, QColor(49, 49, 49))
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(90, 90, 90))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)

    if settings.Theme == "Flashbang":
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(220, 220, 220))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(220, 220, 220))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.Disabled, QPalette.Base, QColor(240, 240, 240))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(160, 160, 160))
        palette.setColor(QPalette.Disabled, QPalette.Button, QColor(230, 230, 230))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(160, 160, 160))
        palette.setColor(QPalette.Disabled, QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(160, 160, 160))
        palette.setColor(
            QPalette.BrightText, Qt.green
        )  # You can choose a suitable bright text color
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(
            QPalette.Highlight, QColor(150, 200, 255)
        )  # Light blue highlight
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor

def set_theme(app):
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
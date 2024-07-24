from PySide6.QtGui import QImage, QPalette, QColor
from PySide6.QtCore import Qt

def styleSheet():
    return (
        "\n"
        "\n"
        "QMainWindow {\n"
        "   color:black;\n"
        "   background-color:black;\n"
        "border:1px;\n"
        "}\n"
        "QLabel{\n"
        "	color: #fff;\n"
        "}\n"
        "QLineEdit{\n"
        "color: #fff;\n"
        "}\n"
        "#centralwidget{\n"
        "	background-color:#1f232a;\n"
        "}\n"
        "#leftMenuSubContainer{\n"
        "	background-color:#16191d;\n"
        "   border-radius: 30px;\n"
        "}\n"
        "#bottomMenuSubContainer{\n"
        "	background-color:#16191d;\n"
        "   border-radius: 30px;\n"
        "}\n"
        "\n"
        "QProgressBar{\n"
        "    background-color:#2c313c;\n"
        "	text-align:left;\n"
        "	padding:5px 10px;\n"
        "	border-radius: 10px;\n"
        "\n"
        "}\n"
        " QProgressBar::chunk {\n"
        "     background-color:white;\n"
        "	text-align:left;\n"
        "	border-radius: 10px;\n"
        " }\n"
        "QStackedWidget{\n"
        "	background-color:#16191d;\n"
        "	text-align:left;\n"
        "	padding:5px 10px;\n"
        "	border-radius: 25px;\n"
        "}\n"
        "QPushButton{\n"
        "    background-color:#2c313c;\n"
        "	text-align:left;\n"
        "	padding:5px 10px;\n"
        "	border-radius: 10px;\n"
        "    color: #fff;\n"
        "}\n"
        "QPushButton:checked{\n"
        "	background-color:#676e7b;\n"
        "}\n"
        "QPushButton:ho"
        "ver{\n"
        "	background-color:#343b47;\n"
        "}\n"
        "\n"
        ""
    )
def Palette():
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(31, 35, 42))
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
    return palette
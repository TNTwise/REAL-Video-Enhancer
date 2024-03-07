import os

homedir = os.path.expanduser(r"~")
if os.path.exists(f"{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer") == False:
    os.system(f'mkdir -p "{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer"')
if os.path.exists(f"{homedir}/.local/share/REAL-Video-Enhancer") == False:
    os.system(f'mkdir -p "{homedir}/.local/share/REAL-Video-Enhancer"')
global flatpak
flatpak = False
if "FLATPAK_ID" in os.environ:
    flatpak = True


def thisdir():
    if flatpak == True:
        return f"{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer"
    return f"{homedir}/.local/share/REAL-Video-Enhancer"

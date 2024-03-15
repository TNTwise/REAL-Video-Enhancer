import os
import platform
homedir = os.path.expanduser(r"~")
os = platform.system()
global flatpak
if os == 'Darwin':
    flatpak = False
    if os.path.exists(f"{homedir}/Library/REAL-Video-Enhancer") == False:
        os.system(f'mkdir -p "{homedir}/Library/REAL-Video-Enhancer"')
if os == 'Linux':
    if os.path.exists(f"{homedir}/.local/share/REAL-Video-Enhancer") == False:
        os.system(f'mkdir -p "{homedir}/.local/share/REAL-Video-Enhancer"')
    
    flatpak = False
    if "FLATPAK_ID" in os.environ:
        flatpak = True


def thisdir():
    if os == 'Linux':
        if flatpak == True:
            if os.path.exists(f"{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer") == False:
                os.system(f'mkdir -p "{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer"')
            return f"{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer"
        return f"{homedir}/.local/share/REAL-Video-Enhancer"
    if os == 'Darwin':
        return f"{homedir}/Library/REAL-Video-Enhancer"

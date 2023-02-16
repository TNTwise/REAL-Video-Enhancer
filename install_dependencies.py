import os
import platform
import distro


linux_distro = distro.id()

class dependencies():
    def install_dependencies():
        if os.path.isfile('/usr/bin/pacman') == True:
            os.system('pkexec pacman -S tk opencv konsole ffmpeg python-pip')
        if os.path.isfile('/usr/bin/apt') == True:
            os.system('pkexec apt install python3-opencv ffmpeg python3-tk konsole libavfilter-dev libavfilter8 libswscale-dev ')
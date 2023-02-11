import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import subprocess
import getpass
passwd = pass_box.get()
    
    
    p = subprocess.Popen((f'echo {passwd} | sudo -S cp "{thisdir}/install/rife-gui" /usr/bin/rife-gui'),shell=TRUE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, error = p.communicate()
    os.system(f'echo {passwd} | sudo -S cp "{thisdir}/icons/Icon.svg" /usr/share/icons/hicolor/scalable/apps/Rife.svg')
    if str(error) != f"b'[sudo] password for {getpass.getuser()}: '":# Add different pop up window here and in other install function that says it completed successfully
        pass_dialog_box_err()
    else:
        os.system(f"echo {passwd} | sudo -S chmod +x /usr/bin/rife-gui")
        passwd=""
        
        os.system(f'cp "{thisdir}/install/Rife-Vulkan-GUI.desktop" /home/$USER/.local/share/applications/')
        os.system("mkdir /home/$USER/Rife-Vulkan-GUI")
        os.system(f"echo {passwd} | sudo -S rm -rf {thisdir}/.git/")
        os.system(f"cp -r * /home/$USER/Rife-Vulkan-GUI")
        os.chdir(f"{thisdir}")
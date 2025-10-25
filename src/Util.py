import os
import warnings
import requests
import stat
import tarfile
import subprocess
import shutil
import platform
import psutil
import cpuinfo
import distro
import webbrowser
import zipfile
import sys
import time
from functools import wraps

from .constants import CWD, HAS_NETWORK_ON_STARTUP, IS_FLATPAK, PLATFORM, HOME_PATH, CPU_ARCH


class FileHandler:
    @staticmethod
    def getFreeSpace() -> int:
        """
        Returns the available disk space in GB.
        """
        try:
            total, used, free = shutil.disk_usage(CWD)
            available_space = free / (1024**3)
            return available_space
        except Exception as e:
            log(f"An error occurred while getting available disk space: {e}")
            return 0
    @staticmethod
    def moveFolder(prev: str, new: str):
        """
        moves a folder from prev to new
        """
        if not os.path.exists(new):
            if not os.path.isfile(new):
                shutil.move(prev, new)
            else:
                print("WARN tried to rename a file to a file that already exists")
        else:
            print("WARN tried to rename a folder to a folder that already exists")

    @staticmethod
    def unzipFile(file, outputDirectory):
        """
        Extracts a zip file in the same directory as the zip file and deletes it after extraction.
        """
        origCWD = os.getcwd()
        dir_path = os.path.dirname(os.path.realpath(file))
        os.chdir(dir_path)
        log("Extracting: " + file)
        with zipfile.ZipFile(file, "r") as f:
            f.extractall(outputDirectory)
        removeFile(file)
        os.chdir(origCWD)

    @staticmethod
    def removeFolder(folder):
        """
        Removes the folder of the current working directory
        """
        if os.path.exists(folder):
            shutil.rmtree(folder)

    @staticmethod
    def removeFile(file):
        """
        Removes the file of the current working directory
        """
        if os.path.isfile(file):
            os.remove(file)

    @staticmethod
    def copy(prev: str, new: str):
        """
        moves a folder from prev to new
        """
        if not os.path.exists(new):
            if not os.path.isfile(new):
                shutil.copytree(prev, new)
            else:
                print("WARN tried to rename a file to a file that already exists")
        else:
            print("WARN tried to rename a folder to a folder that already exists")

    @staticmethod
    def copyFile(prev: str, new: str):
        """
        moves a file from prev to a new directory (new)
        """
        if not os.path.isfile(new):
            shutil.copy(prev, new)
        else:
            print("WARN tried to rename a file to a file that already exists")

    @staticmethod
    def moveFile(prev: str, new: str):
        """
        moves a file from prev to new
        """
        if not os.path.exists(new):
            if not os.path.isfile(new):
                os.rename(prev, new)
            else:
                print("WARN tried to rename a file to a file that already exists")
        else:
            print("WARN tried to rename a folder to a folder that already exists")

    @staticmethod
    def makeExecutable(file_path):
        st = os.stat(file_path)
        os.chmod(file_path, st.st_mode | stat.S_IEXEC)

    @staticmethod
    def createDirectory(dir: str):
        if not os.path.exists(dir):
            os.mkdir(dir)

    @staticmethod
    def getUnusedFileName(base_file_name: str, outputDirectory: str, extension: str):
        """
        Returns an unused file name by adding an iteration number to the file name.
        """
        iteration = 0
        output_file = base_file_name
        while os.path.isfile(base_file_name):
            output_file = os.path.join(
                outputDirectory,
                f"{base_file_name}_({iteration}).{extension}",
            )
            iteration += 1
        return output_file
    @staticmethod
    def getDefaultOutputFolder() -> str:
        """
        Returns the default output folder based on the operating system.
        """
        videos_folder = os.path.join(f"{HOME_PATH}", "Videos")
        
        if PLATFORM == "linux":
            try:
                result = subprocess.run(
                    ["xdg-user-dir", "VIDEOS"], capture_output=True, text=True
                ).stdout.strip()
                if os.path.isdir(result):
                    videos_folder = result
            except Exception as e:
                log(f"An error occurred while getting the Videos folder on Linux: {e}")

        if PLATFORM == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                CSIDL_MYVIDEO = 0x000e
                SHGFP_TYPE_CURRENT = 0

                buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
                ctypes.windll.shell32.SHGetFolderPathW(
                    None, CSIDL_MYVIDEO, None, SHGFP_TYPE_CURRENT, buf
                )
                if os.path.isdir(buf.value):
                    videos_folder = buf.value
            except Exception as e:
                log(f"An error occurred while getting the Videos folder on Windows: {e}")

        if PLATFORM == "darwin":
            videos_folder = os.path.join(f"{HOME_PATH}", "Desktop")

        return videos_folder

def log(message: str):
    
    try:
        with open(os.path.join(CWD, "log.txt"), "a") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"An error occurred while logging: {e}", file=sys.stderr)


with open(os.path.join(CWD, "log.txt"), "w") as f:
    pass




def getAvailableDiskSpace() -> float:
    """
    Returns the available disk space in GB.
    """
    try:
        total, used, free = shutil.disk_usage(CWD)
        available_space = free / (1024**3)
        return available_space
    except Exception as e:
        log(f"An error occurred while getting available disk space: {e}")
        return "Unknown"


def networkCheck(hostname="https://raw.githubusercontent.com") -> bool:
    """
    checks network availability against a url, default url: raw.githubusercontent.com
    """
    try:
        _ = requests.head(hostname, timeout=1)
        return True
    except Exception as e:
        log(str(e))
        log("No internet connection available.")
    return False


def getOSInfo() -> str:
    try:
        """
        Returns the exact name of the operating system along with additional information like 64-bit.
        """
        system = platform.system()
        release = platform.release()
        architecture = platform.machine()
        if system == "Linux":
            distro_name = distro.name()
            distro_version = distro.version()
            return f"{distro_name} {distro_version} {architecture}"
        return f"{system} {release} {architecture}"
    except Exception as e:
        log(f"An error occurred while getting OS information: {e}")
        return "Unknown"


def getRAMAmount() -> str:
    """
    Returns the amount of RAM in the system.
    """
    try:
        ram = psutil.virtual_memory().total
        ram_gb = ram / (1024**3)
        return f"{ram_gb:.2f} GB"
    except Exception as e:
        log(f"An error occurred while getting RAM amount: {e}")
        return "Unknown"


def removeFolder(folder):
    """
    Removes the folder of the current working directory
    """
    shutil.rmtree(folder)


def getCPUInfo() -> str:
    """
    Returns the CPU information of the system.
    """
    # return platform.processor() + " " + str(psutil.cpu_count(logical=False)) + " cores" + platform.
    try:
        if PLATFORM == "win32":
            try:
                # Run the 'wmic' command to get CPU information
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Split the result by lines and return the second line which contains the CPU name
                return result.stdout.split("\n")[2].strip()
            except Exception as e:
                print(f"An error occurred while getting CPU brand: {e}")
                return "X86_64 CPU" if CPU_ARCH == "x86_64" else "ARM64 CPU"
        else:
            return cpuinfo.get_cpu_info()["brand_raw"]
    except Exception as e:
        log(f"An error occurred while getting CPU information: {e}")
        return "Unknown"


def copy(prev: str, new: str):
    """
    moves a folder from prev to new
    """
    if not os.path.exists(new):
        if not os.path.isfile(new):
            shutil.copytree(prev, new)
        else:
            print("WARN tried to rename a file to a file that already exists")
    else:
        print("WARN tried to rename a folder to a folder that already exists")


def copyFile(prev: str, new: str):
    """
    moves a file from prev to a new directory (new)
    """
    if not os.path.isfile(new):
        shutil.copy(prev, new)
    else:
        print("WARN tried to rename a file to a file that already exists")


def move(prev: str, new: str):
    """
    moves a file from prev to new
    """
    if not os.path.exists(new):
        if not os.path.isfile(new):
            os.rename(prev, new)
        else:
            print("WARN tried to rename a file to a file that already exists")
    else:
        print("WARN tried to rename a folder to a folder that already exists")


def makeExecutable(file_path):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)


def warnAndLog(message: str):
    warnings.warn(message)
    log("WARN: " + message)


def createDirectory(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)


def currentDirectory():
    return CWD


def removeFile(file):
    try:
        os.remove(file)
    except Exception:
        print("Failed to remove file!")


def downloadFile(link, downloadLocation):
    response = requests.get(
        link,
        stream=True,
    )

    with open(downloadLocation, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)


def extractTarGZ(file):
    """
    Extracts a tar gz in the same directory as the tar file and deleted it after extraction.
    """
    origCWD = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(file))
    os.chdir(dir_path)
    log("Extracting: " + file)
    with tarfile.open(file, "r:gz") as f:
        f.extractall()
    removeFile(file)
    os.chdir(origCWD)


def get_gpu_info():
    if PLATFORM == "win32":
        try:
            output = subprocess.check_output(
                "wmic path win32_VideoController get name", shell=True
            ).decode()
            return output.strip().split("\n")[1]
        except Exception:
            return "Unable to retrieve GPU info on Windows"

    elif PLATFORM == "darwin":  # macOS
        try:
            output = subprocess.check_output(
                "system_profiler SPDisplaysDataType | grep Vendor", shell=True
            ).decode()
            return output.strip().split(":")[1].strip()
        except Exception:
            return "Unable to retrieve GPU info on macOS"

    elif PLATFORM == "linux":
        try:
            # Try lspci command first
            output = subprocess.check_output("lspci | grep -i vga", shell=True).decode()
            return output.strip().split(":")[2].strip()
        except Exception:
            try:
                # If lspci fails, try reading from /sys/class/graphics
                with open("/sys/class/graphics/fb0/device/vendor", "r") as f:
                    vendor_id = f.read().strip()
                return f"Vendor ID: {vendor_id}"
            except Exception:
                return "Unable to retrieve GPU info on Linux"

    else:
        return "Unsupported operating system"


def getVendor():
    """
    Gets GPU vendor of the system
    vendors = ["Intel", "AMD", "Nvidia"]
    """
    gpuInfo = get_gpu_info()
    vendors = ["Intel", "AMD", "Nvidia"]
    for vendor in vendors:
        if vendor.lower() in gpuInfo.lower():
            return vendor


def openLink(link: str):
    """
    Opens a link in the default web browser.

    :param link: The link to open.
    :type link: str
    """
    webbrowser.open(link)


def errorAndLog(message: str):
    log("ERROR: " + message)
    raise os.error("ERROR: " + message)


def checkForWritePermissions(dir):
    """
    Checks for write permissions in the current directory.

    Also reads the flatpak-info file to see if the directory is in the current allowed r/w dirs.
    Args:
        - the directory to check if permissions are in
    """
    if not os.path.isdir(dir):
        return False

    i = 2  # change this to 1 to debug flatpak
    if IS_FLATPAK or i == 1:
        with open("/.flatpak-info", "r") as f:
            result = f.readlines()

        directories_with_permissions = []
        for i in result:
            if "filesystems=" in i:
                i = i.split(";")
                s = []
                for e in i:
                    if len(e) > 0 and i != "\n":
                        s.append(e)
                for j in s:
                    j = j.replace("filesystems=", "")
                    if j == "xdg-download":
                        j = f"{HOME_PATH}/Downloads"
                    j = j.replace("xdg-", f"{HOME_PATH}/")
                    j = j.replace("~", f"{HOME_PATH}")
                    directories_with_permissions.append(j)
        for i in directories_with_permissions:
            if dir[-1] != "/":
                dir += "/"
            log(
                f"Checking dir: {i.lower()} is in or equal to Selected Dir: {dir.lower()}"
            )

            if (
                i.lower() in dir.lower()
                or "io.github.tntwise.real-video-enhancer" in dir.lower()
                and ":ro" not in i
            ):
                return True
            else:
                if "/run/user/1000/doc/" in dir:
                    dir = dir.replace("/run/user/1000/doc/", "").split("/")
                    permissions_dir = ""
                    for index in range(len(dir)):
                        if index != 0:
                            permissions_dir += f"{dir[index]}/"
                    if HOME_PATH not in permissions_dir:
                        dir = f"{HOME_PATH}/{permissions_dir}"
                    else:
                        dir = f"/{permissions_dir}"

                log(
                    f"Checking dir: {i.lower()} is in or equal to Selected Dir: {dir.lower()}"
                )
                if (
                    i.lower() in dir.lower()
                    or "io.github.tntwise.real-video-enhancer" in dir.lower()
                    and ":ro" not in i
                ):
                    return True

        return False
    else:
        if os.access(dir, os.R_OK) and os.access(dir, os.W_OK):
            return True
        return False

def open_folder(folder):
    if PLATFORM == "win32":
        os.startfile(folder)
    elif PLATFORM == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])


class subprocess_popen_without_terminal(subprocess.Popen):
    """
    A class that allows you to run a subprocess without opening a terminal window.
    """
    def __init__(self, *args, **kwargs):
        if PLATFORM == "win32":
                kwargs["startupinfo"] = subprocess.STARTUPINFO()
                kwargs["startupinfo"].dwFlags |= subprocess.STARTF_USESHOWWINDOW
        super().__init__(*args, **kwargs)
    

def create_independent_process(target_func, *args, **kwargs):
    """
    Creates a completely independent process that won't interfere with PyQt's main application
    """
    import os
    import sys
    from multiprocessing import Process, set_start_method

    # Force the 'spawn' method on all platforms for complete process isolation
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set, ignore
        pass
    
    # Set environment variables to avoid Qt conflicts
    env = os.environ.copy()
    env['QT_PLUGIN_PATH'] = ''
    env['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    
    # Create process with isolated environment
    process = Process(target=target_func, args=args, kwargs=kwargs)
    
    # Set process to daemon so it terminates with main process
    process.daemon = True
    
    return process

def print_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} : {end - start:.6f} seconds")
        return result
    return wrapper


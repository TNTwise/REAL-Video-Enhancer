import os
import stat
import zipfile
import shutil
from .Util import log

class FileHandler:
    @staticmethod
    def getFreeSpace() -> int:
        """
        Returns the available disk space in GB.
        """
        try:
            total, used, free = shutil.disk_usage(os.getcwd())
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
        FileHandler.removeFile(file)
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
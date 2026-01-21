import os
import pathlib
import shutil
import stat
import zipfile

from .LogConfig import get_logger

logger = get_logger(__name__)


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
        except Exception:
            logger.exception(
                'An error occurred while getting available disk space'
            )
            return 0

    @staticmethod
    def moveFolder(prev: str, new: str):
        """
        Moves a folder from prev to new
        """
        if not pathlib.Path(new).exists():
            if not pathlib.Path(new).is_file():
                shutil.move(prev, new)
            else:
                print(
                    'WARN tried to rename a file to a file that already exists'
                )
        else:
            print(
                'WARN tried to rename a folder to a folder that already exists'
            )

    @staticmethod
    def unzipFile(file, outputDirectory):
        """
        Extracts a zip file in the same directory as the zip file and deletes it after extraction.
        """
        origCWD = os.getcwd()
        dir_path = os.path.dirname(os.path.realpath(file))
        os.chdir(dir_path)
        logger.info('Extracting: %s', file)
        with zipfile.ZipFile(file, 'r') as f:
            f.extractall(outputDirectory)
        FileHandler.removeFile(file)
        os.chdir(origCWD)

    @staticmethod
    def removeFolder(folder):
        """
        Removes the folder of the current working directory
        """
        if pathlib.Path(folder).exists():
            shutil.rmtree(folder)

    @staticmethod
    def removeFile(file):
        """
        Removes the file of the current working directory
        """
        if pathlib.Path(file).is_file():
            pathlib.Path(file).unlink()

    @staticmethod
    def copy(prev: str, new: str):
        """
        Moves a folder from prev to new
        """
        if not pathlib.Path(new).exists():
            if not pathlib.Path(new).is_file():
                shutil.copytree(prev, new)
            else:
                print(
                    'WARN tried to rename a file to a file that already exists'
                )
        else:
            print(
                'WARN tried to rename a folder to a folder that already exists'
            )

    @staticmethod
    def copyFile(prev: str, new: str):
        """
        Moves a file from prev to a new directory (new)
        """
        if not pathlib.Path(new).is_file():
            shutil.copy(prev, new)
        else:
            print('WARN tried to rename a file to a file that already exists')

    @staticmethod
    def moveFile(prev: str, new: str):
        """
        Moves a file from prev to new
        """
        if not pathlib.Path(new).exists():
            if not pathlib.Path(new).is_file():
                pathlib.Path(prev).rename(new)
            else:
                print(
                    'WARN tried to rename a file to a file that already exists'
                )
        else:
            print(
                'WARN tried to rename a folder to a folder that already exists'
            )

    @staticmethod
    def makeExecutable(file_path):
        st = os.stat(file_path)
        pathlib.Path(file_path).chmod(st.st_mode | stat.S_IEXEC)

    @staticmethod
    def createDirectory(dir: str):
        if not pathlib.Path(dir).exists():
            pathlib.Path(dir).mkdir()

    @staticmethod
    def getUnusedFileName(
        base_file_name: str, outputDirectory: str, extension: str
    ):
        """
        Returns an unused file name by adding an iteration number to the file name.
        """
        iteration = 0
        output_file = base_file_name
        while pathlib.Path(base_file_name).is_file():
            output_file = os.path.join(
                outputDirectory,
                f'{base_file_name}_({iteration}).{extension}',
            )
            iteration += 1
        return output_file

import os
import csv
import src.programData.thisdir

thisdir = src.programData.thisdir.thisdir()
from src.programData.return_data import *
from src.misc.messages import *

homedir = os.path.expanduser(r"~")
from src.misc.log import *
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QInputDialog,
    QLineEdit,
    QFileDialog,
    QMessageBox,
)
from src.programData.write_permisions import *
import traceback
from sys import exit


# im going to eventually redo this
class CustomException(Exception):
    def __init__(self, additional_info):
        self.additional_info = additional_info
        super().__init__()

    def __str__(self):
        return f"{super().__str__()} - Additional Info: {self.additional_info}"


class Settings:
    def __init__(self) -> None:
        ManageFiles.create_folder(f"{thisdir}/files")
        ManageFiles.create_file(f"{thisdir}/files/settings.txt")
        try:
            self.readSettings()
        except Exception as e:
            tb = traceback.format_exc()
            tb = tb.split("\n")[-10:]

            log(f"RECURSION OVERFLOW!!! {tb}{e}")
            exit()

    def write_to_settings_file(self, description, option):
        with open(f"{thisdir}/files/settings.txt", "a") as f:
            f.write(description + "," + option + "\n")

    def readSettings(self):
        settings_dict = {}
        with open(f"{thisdir}/files/settings.txt", "r") as f:
            f = csv.reader(f)
            for row in f:
                try:
                    settings_dict[row[0]] = row[1]
                except:
                    pass
        default_settings = {
            "FixFFMpegCatchup": "Disabled",
            "Image_Type": ".jpg",
            "videoQuality": "18",
            "FrameIncrements": "100",
            "Theme": "Dark",
            "GPUUsage": "Default",
            "ModelDir": f"{thisdir}/models/",
            "RenderType": "Optimized",
            "SceneChangeDetectionMode": "Enabled",
            "RenderDir": f"{thisdir}/renders/",
            "ExtractionImageType": "jpg",
            "SceneChangeDetection": "0.3",
            "SceneChangeMethod": "ffmpeg",
            "Encoder": "264",
            "DiscordRPC": "Enabled",
            "FrameIncrementsMode": "Automatic",
            "UpdateChannel": "Stable",
            "DefaultRifeModel": f"rife-v4.15",
            "ignoreVramPopup": "False",
            "Notifications": "Enabled",
            "UHDResCutOff": "1080",
            "gpuID": "0",
            "HalfPrecision": "False",
        }
        if returnOperatingSystem() == "Linux":
            default_settings["OutputDir"] = f"{homedir}/Videos/"
        if returnOperatingSystem() == "MacOS":
            default_settings["OutputDir"] = f"{homedir}/Desktop/"
        for setting, default_value in default_settings.items():
            try:
                setattr(self, setting, settings_dict[setting])
                if setting in ["OutputDir", "RenderDir"] and not os.path.exists(
                    getattr(self, setting)
                ):
                    log(
                        f"This most likely means the output directory does not exist, in which create {homedir}/Videos, or you do not have permission to output there.\nEither set the output directory {homedir}/Videos or allow permission for the new directory."
                    )
                    self.write_to_settings_file(setting, default_value)
                    raise CustomException(
                        f"This most likely means the output directory does not exist, in which create {homedir}/Videos, or you do not have permission to output there.\nEither set the output directory {homedir}/Videos or allow permission for the new directory."
                    )
                if setting == "FrameIncrements":
                    self.FrameIncrements = int(self.FrameIncrements)

            except Exception as e:
                log(e)
                self.write_to_settings_file(setting, default_value)

                self.readSettings()

        try:
            self.VRAM = settings_dict["VRAM"]
        except:
            self.write_to_settings_file("VRAM", f"2")

            self.readSettings()

    def change_setting(self, setting, svalue):
        original_settings = {}
        with open(f"{thisdir}/files/settings.txt", "r") as f:
            f = csv.reader(f)
            for row in f:
                original_settings[row[0]] = row[1]

            original_settings[setting] = svalue
            os.remove(f"{thisdir}/files/settings.txt")
            for key, value in original_settings.items():
                with open(f"{thisdir}/files/settings.txt", "a") as f:
                    f.write(key + "," + value + "\n")
        self.readSettings()

    def is_setting(setting):
        with open(f"{thisdir}/files/settings.txt", "r") as f:
            f = csv.reader(f)
            for row in f:
                if row[0] == setting:
                    return True
            return False


def change_UHD_res(self):
    settings = Settings()
    settings.change_setting("UHDResCutOff", str(self.ui.UHDResCutoffSpinBox.value()))


def change_GPU_ID(self):
    settings = Settings()
    settings.change_setting("gpuID", str(self.ui.gpuIDSpinBox.value()))


def changeDiscordRPC(self):
    settings = Settings()
    if self.ui.DiscordRPCBox.isChecked() == True:
        settings.change_setting("DiscordRPC", f"Enabled")
    else:
        settings.change_setting("DiscordRPC", f"Disabled")


def changeSceneDetection(self):
    settings = Settings()
    if (
        len(self.ui.sceneChangeLineEdit.text()) > 0
        and int(self.ui.sceneChangeLineEdit.text()) != 0
    ):
        settings.change_setting(
            "SceneChangeDetection", f"0.{self.ui.sceneChangeLineEdit.text()}"
        )


def selRenderDir(self):
    self.settings = Settings()
    render_folder = QFileDialog.getExistingDirectory(self, "Open Folder")

    if render_folder != "":
        if check_for_write_permissions(render_folder):
            try:
                os.mkdir(f"{render_folder}/renders/")
                self.settings.change_setting("RenderDir", f"{render_folder}/renders/")

                self.ui.RenderPathLabel.setText(f"{self.settings.RenderDir}")
            except:
                if already_Render_folder(self):
                    self.settings.change_setting(
                        "RenderDir", f"{render_folder}/renders/"
                    )

                    self.ui.RenderPathLabel.setText(f"{self.settings.RenderDir}")
                else:
                    pass

        else:
            no_perms_change_setting(self)
    self.settings = Settings()


def selOutputDir(self):
    settings = Settings()
    output_folder = QFileDialog.getExistingDirectory(self, "Open Folder")
    if output_folder != "":
        if check_for_write_permissions(output_folder):
            self.output_folder = output_folder
            settings.change_setting("OutputDir", f"{self.output_folder}")

            self.ui.OutputDirectoryLabel.setText(f"{settings.OutputDir}")
        else:
            no_perms_change_setting(self)
    self.settings = Settings()


def selEncoder(self):
    settings = Settings()
    if self.ui.EncoderCombo.currentText() == "Lossless":
        transition_detection_lossless_warning(self)
    settings.change_setting("Encoder", self.ui.EncoderCombo.currentText())
    self.encoder = settings.Encoder
    self.settings = Settings()


def selRenderType(self):
    settings = Settings()
    if "Classic" in self.ui.renderTypeCombo.currentText():
        settings.change_setting("RenderType", "Classic")
    if "Optimized" == self.ui.renderTypeCombo.currentText():
        settings.change_setting("RenderType", "Optimized")
    if "Optimized (Incremental)" == self.ui.renderTypeCombo.currentText():
        settings.change_setting("RenderType", "Optimized (Incremental)")


def selFrameIncrements(value):
    settings = Settings()
    settings.change_setting("FrameIncrements", f"{value}")


def selVidQuality(self):
    settings = Settings()
    if self.ui.VidQualityCombo.currentText() == "Lossless":
        settings.change_setting("videoQuality", "10")
    if self.ui.VidQualityCombo.currentText() == "Very High":
        settings.change_setting("videoQuality", "14")
    if self.ui.VidQualityCombo.currentText() == "High":
        settings.change_setting("videoQuality", "18")
    if self.ui.VidQualityCombo.currentText() == "Medium":
        settings.change_setting("videoQuality", "20")
    if self.ui.VidQualityCombo.currentText() == "Low":
        settings.change_setting("videoQuality", "22")
    self.videoQuality = settings.videoQuality


def selFrameIncrementsMode(self):
    settings = Settings()
    settings.change_setting(
        "FrameIncrementsMode", self.ui.frameIncrementsModeCombo.currentText()
    )
    if settings.FrameIncrementsMode == "Automatic":
        self.ui.frameIncrementSpinBox.hide()
        self.ui.label_7.hide()
    else:
        self.ui.frameIncrementSpinBox.show()
        self.ui.label_7.show()


def selSceneDetectionMode(self):
    settings = Settings()
    if self.ui.sceneChangeDetectionCheckBox.isChecked() == True:
        settings.change_setting("SceneChangeDetectionMode", "Enabled")
        self.ui.label_3.show()
        self.ui.sceneChangeSensativityButton.show()
        self.ui.sceneChangeLineEdit.show()
        self.ui.label_17.show()
        self.ui.sceneChangeMethodComboBox.show()
    else:
        settings.change_setting("SceneChangeDetectionMode", "Disabled")
        self.ui.label_3.hide()
        self.ui.sceneChangeSensativityButton.hide()
        self.ui.sceneChangeLineEdit.hide()
        self.ui.label_17.hide()
        self.ui.sceneChangeMethodComboBox.hide()


def selSceneDetectionMethod(self):
    settings = Settings()
    settings.change_setting(
        "SceneChangeMethod", self.ui.sceneChangeMethodComboBox.currentText()
    )


def Notifications(self):
    settings = Settings()
    if self.ui.NotificationsCheckBox.isChecked() == True:
        settings.change_setting("Notifications", "Enabled")
    else:
        settings.change_setting("Notifications", "Disabled")


def halfPrecision(self):
    settings = Settings()
    settings.change_setting(
        "HalfPrecision", f"{self.ui.halfPrecisionCheckBox.isChecked()}"
    )


def uninstallAPP(self):
    if uninstallMessage(self):
        os.system(f'rm -rf "{thisdir}/"*')
        exit()

#!/usr/bin/python3

from src.misc.log import log
import os
from src.misc.createDirectories import createDirectories, createFiles
from src.programData.settings import *

createDirectories()
createFiles()
import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QListWidget,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QGraphicsOpacityEffect,
)
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QBrush
from PyQt5.QtCore import (
    Qt,
    QSize,
    QEvent,
    QThread,
    QPropertyAnimation,
    QEasingCurve,
    QRect,
)
from rife_ncnn_vulkan_python import Rife
import rife_ncnn_vulkan_python.rife_ncnn_vulkan_wrapper
from src.programData.version import returnVersion

setHIPGFXVersion()

try:
    import torch
    import torchvision
    import spandrel
    from spandrel import ModelLoader
    from modules.handelModel import handleModel

    torch_version = True
    log("torch_version")

except Exception as e:
    log(f"ncnn_verson {e}")
    torch_version = False
try:
    import cupy
    import modules.GMFSSCUDA as GMFSSCUDA

    gmfss = True
except:
    gmfss = False
if torch_version:
    import modules.RifeCUDA as rifeCUDA
    import modules.RealESRGANCUDA as RealESRGANCUDA
    import modules.CustomModelsCUDA as CustomModelsCUDA

    import numpy as np

from upscale_ncnn_py import UPSCALE

homedir = os.path.expanduser(r"~")
try:
    os.system(f'mkdir -p "{homedir}/Videos/"')
except:
    pass
import src.programData.thisdir

import src.programData.checks as checks

thisdir = src.programData.thisdir.thisdir()
if os.path.exists(f"{thisdir}") == False:
    os.mkdir(f"{thisdir}")


import src.programData.theme as theme
import traceback

import src.getModels.select_models as sel_mod
import src.getModels.get_models_settings
import sys
import psutil

import mainwindow
import os
from src.programData.write_permisions import *
from threading import *
from src.programData.return_data import *

ManageFiles.create_folder(f"{thisdir}/files/")
import src.runAI.workers as workers
import time

# import src.get_models as get_models
from time import sleep
from multiprocessing import cpu_count
from src.misc.messages import *
import modules.Rife as rife
import modules.ESRGAN as esrgan
import modules.Waifu2X as Waifu2X
import modules.IFRNET as ifrnet
import modules.CUGAN as cugan
import modules.realsr as realsr
import modules.VapoursynthRifeNCNN as VapoursynthRifeNCNN
import modules.CustomModelsNCNN as CustomModelsNCNN
import modules.SPANNCNN as span


import src.misc.onProgramStart
from src.misc.getNCNNScale import returnScale
from src.runAI.ETA import *
from src.getLinkVideo.get_video import *


from src.programData.return_data import *
from src.programData.checks import *
from src.programData.return_latest_update import *


class FileDropWidget(QLabel):
    def __init__(self, parent=None):
        super(FileDropWidget, self).__init__(parent)
        self.main = parent
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.add_file_item(file_path)

    def add_file_item(self, file_path):
        item = QListWidgetItem(file_path)
        try:
            cap = cv2.VideoCapture(file_path)
            # Check if the file can be opened successfully
            if cap.isOpened():
                log(f"{file_path} is a video file.")
                cap.release()
                # success!
                self.main.input_file = item.text()

                self.main.download_youtube_video_command = ""
                self.main.localFile = True
                self.main.videoName = VideoName.return_video_name(
                    f"{self.main.input_file}"
                )
                if '"' in self.main.input_file:
                    quotes(self.main)
                    self.main.input_file = ""
                else:
                    self.main.showChangeInFPS()
                    self.main.ui.logsPreview.clear()
                    self.main.addLinetoLogs(f"Input file = {item.text()}")
            else:
                log(f"{file_path} is not a video file.")

                not_a_video(self.main)
        except Exception as e:
            if check_if_flatpak():
                self.main.showDialogBox(
                    str(e)
                    + f"\nMost likely no permissions to access this directory in flatpak.\nEither select a file from the Input Video button\nOr drag a file from a directory with permissions, most likely {homedir}/Videos/."
                )
            else:
                self.main.showDialogBox(str(e))
            traceback_info = traceback.format_exc()
            log(f"{e} {traceback_info}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        self.aspect_ratio = 16 / 9
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setMinimumSize(1000, 550)
        self.resize(1000, 550)
        self.on = True
        self.localFile = True
        self.input_file = ""
        self.output_folder = ""
        self.download_youtube_video_command = ""
        self.benchmark = False
        for i in sys.argv:  # parse args
            if "--benchmark" == i:
                self.benchmark = True

        # self.ui.logsPreview.setStyleSheet("color: white; background-color: rgb(32,28,28); border-radius: 10px;")
        # self.ui.imagePreview.setStyleSheet("border-radius: 10px;")
        # self.fadeIn(self.ui.verticalTabWidget) # < issues with qtextedit, adding in later
        self.settings = Settings()
        self.thread2 = QThread()
        # Step 3: Create a worker object
        worker = return_latest()

        # Step 4: Move worker to the thread
        worker.moveToThread(self.thread2)
        # Step 5: Connect signals and slots
        self.thread2.started.connect(worker.run)
        worker.finished.connect(self.thread2.quit)
        worker.finished.connect(worker.deleteLater)
        self.thread2.finished.connect(self.thread2.deleteLater)
        worker.progress.connect(self.addVersionstoLogs)
        # Step 6: Start the thread

        self.thread2.start()
        self.addLinetoLogs(f"Current Version: {returnVersion()}")

        src.misc.onProgramStart.onApplicationStart(self)
        src.misc.onProgramStart.bindButtons(self)

        self.file_drop_widget = FileDropWidget(self)
        self.ui.verticalTabWidget.setFocusPolicy(Qt.ClickFocus)
        self.pixmap = QPixmap(f"{thisdir}/icons/Dragndrop.png")
        scaled_pixmap = self.pixmap.scaled(
            self.size() / 2, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.file_drop_widget.setPixmap(scaled_pixmap)

        # self.label.setPixmap(self.pixmap)
        # self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # self.file_drop_widget.setScaledContents(True)
        self.file_drop_widget.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.ui.imageFormLayout.addWidget(self.file_drop_widget)

        self.ui.frameIncrementsModeCombo.setCurrentText(
            self.settings.FrameIncrementsMode
        )

        self.ui.installModelsProgressBar.setMaximum(100)

        selFrameIncrementsMode(self)

        if check_for_write_permissions(self.settings.OutputDir) == False:
            no_perms(self)
            try:
                os.mkdir(f"{homedir}/Videos/")
            except:
                pass
            if check_for_write_permissions(f"{homedir}/Videos/"):
                self.settings.change_setting("OutputDir", f"{homedir}/Videos/")
            else:
                no_perms_anywhere(self, self.settings.OutputDir)
        if check_for_write_permissions(self.settings.RenderDir) == False:
            no_perms(self)
            try:
                os.mkdir(f"{thisdir}/renders/")
            except:
                pass
            if check_for_write_permissions(f"{thisdir}/renders/"):
                self.settings.change_setting("RenderDir", f"{thisdir}/renders/")
            else:
                no_perms_anywhere(self, settings.RenderDir)

        self.show()

    def set_default_background_color(self, widget):
        # Get the default background color
        default_background_color = widget.palette().color(widget.backgroundRole())

        # Set the background color to its default value
        widget.setStyleSheet(
            f"background-color: rgb({default_background_color.red()}, {default_background_color.green()}, {default_background_color.blue()});"
        )

    def resizeEvent(self, event):
        # Resize the pixmap to maintain the aspect ratio
        try:
            scaled_pixmap = self.pixmap.scaled(
                self.size() / 2, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.file_drop_widget.setPixmap(scaled_pixmap)
        except Exception as e:
            log("ResizeEvent FAILED!")
            log(f"resizeEvent failed! {e}")

    def restore_default_settings(self):
        with open(f"{thisdir}/files/settings.txt", "w") as f:
            pass
        src.misc.onProgramStart.onApplicationStart(self)
        self.ui.verticalTabWidget.setCurrentIndex(self.ui.verticalTabWidget.count())
        self.ui.SettingsMenus.setCurrentRow(0)
        self.ui.GeneralOptionsFrame.show()

    def addVersionstoLogs(self, n):
        self.addLinetoLogs(f"Latest Stable: {n[1]} Latest Beta: {n[0]}")

    def changeVRAM(self):
        self.settings.change_setting("VRAM", f"{self.ui.gpuThreadingSpinBox.value()}")
        self.gpuMemory = self.settings.VRAM

    def setDirectories(self):
        self.models_dir = f"{thisdir}/models/"

    def switchUI(self):
        if self.ui.AICombo.currentText() == "Rife (NCNN)":
            rife.modelOptions(self)

        if self.ui.AICombo.currentText() == "RealESRGAN (NCNN)":
            esrgan.modelOptions(self)

        if self.ui.AICombo.currentText() == "Waifu2X (NCNN)":
            Waifu2X.modelOptions(self)

        if self.ui.AICombo.currentText() == "Vapoursynth-RIFE":
            VapoursynthRifeNCNN.modelOptions(self)

        if self.ui.AICombo.currentText() == "IFRNET (NCNN)":
            ifrnet.modelOptions(self)

        if self.ui.AICombo.currentText() == "RealCUGAN (NCNN)":
            cugan.modelOptions(self)

        if self.ui.AICombo.currentText() == "RealSR (NCNN)":
            realsr.modelOptions(self)

        if self.ui.AICombo.currentText() == "Custom NCNN models":
            CustomModelsNCNN.modelOptions(self)

        if self.ui.AICombo.currentText() == "GMFSS CUDA (Nvidia only)":
            GMFSSCUDA.modelOptions(self)

        if self.ui.AICombo.currentText() == "Rife CUDA/ROCm (Nvidia/AMD only)":
            rifeCUDA.modelOptions(self)

        if self.ui.AICombo.currentText() == "Rife TensorRT (Nvidia only)":
            rifeCUDA.modelOptions(self, trt=True)

        if self.ui.AICombo.currentText() == "RealESRGAN CUDA/ROCm (Nvidia/AMD only)":
            RealESRGANCUDA.modelOptions(self)

        if self.ui.AICombo.currentText() == "RealESRGAN TensorRT (Nvidia only)":
            RealESRGANCUDA.modelOptions(self, trt=True)

        if self.ui.AICombo.currentText() == "Custom CUDA/ROCm models (Nvidia/AMD only)":
            CustomModelsCUDA.modelOptions(self)

        if self.ui.AICombo.currentText() == "Custom TensorRT models":
            CustomModelsCUDA.modelOptions(self, trt=True)

        if self.ui.AICombo.currentText() == "SPAN (NCNN)":
            span.modelOptions(self)

    def switchMode(self):
        self.ui.AICombo.clear()
        for key, value in self.model_labels.items():
            if value == self.ui.modeCombo.currentText().lower():
                self.ui.AICombo.addItem(key)

    def get_models_from_dir(self, AI):
        return_list = []

        for i in os.listdir(f"{settings.ModelDir}/{AI.lower()}"):
            if os.path.isfile(f"{settings.ModelDir}/{AI.lower()}/{i}") == False:
                return_list.append(f"{i}")  # Adds model to GUI.

        return return_list

    def get_pid(self, name):
        p = psutil.process_iter(attrs=["pid", "name"])
        for process in p:
            if process.info["name"] == name:
                pid = process.info["pid"]

                return pid

    def fadeIn(self, widget, duration=250):
        opacity_effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(opacity_effect)

        self.animation = QPropertyAnimation(opacity_effect, b"opacity")
        self.animation.setDuration(duration)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()

    def resume_render(self):
        self.ui.RifeResume.hide()  # show resume button

        # Thread(target=lambda: Rife(self,(self.ui.Rife_Model.currentText().lower()),2,self.input_file,self.output_folder,1)).start()
        self.ui.RifePause.show()

    def videoProperties(self):
        self.amountFrames = VideoName.return_video_frame_count(self.input_file)
        self.fps = VideoName.return_video_framerate(self.input_file)

    def showChangeInFPS(self, fps=None):
        try:
            width = int(self.ytVidRes.split("x")[0])
            height = int(self.ytVidRes.split("x")[1].replace(" (Enhanced bitrate)", ""))
        except Exception as e:
            log(e)
            try:
                if self.localFile == True:
                    resolution = VideoName.return_video_resolution(self.input_file)
                    width = int(resolution[0])
                    height = int(resolution[1])
                    if width > 3840 or height > 2160:
                        too_large_video(self)
            except:
                log("Couldnt grab resolution in showChangeInFPS")

        try:
            if self.localFile == True:
                self.videoProperties()

            fps = self.fps

            if self.render == "rife":
                try:
                    self.times = int(self.ui.Rife_Times.currentText()[0])

                    if fps != None:
                        if (
                            self.ui.AICombo.currentText() != "Rife"
                            or "v4" not in self.ui.Rife_Model.currentText()
                        ):
                            self.ui.FPSPreview.setText(
                                f"FPS: {(round(fps))} -> {round(fps*self.times)}"
                            )
                        if (
                            "Rife" in self.ui.AICombo.currentText()
                            and "v4" in self.ui.Rife_Model.currentText()
                            and "cuda" not in self.ui.AICombo.currentText().lower()
                        ):
                            self.ui.FPSPreview.setText(f"FPS:")
                            self.ui.FPSFrom.setMinimum(fps)
                            self.ui.FPSFrom.setValue(fps)

                            self.ui.FPSFrom.setMaximum(fps)
                            self.ui.FPSTo.setMinimum(fps * 2)

                            self.ui.FPSTo.setValue((fps) * int(self.times))
                            # log((self.amountFrames / self.ui.FPSFrom.value()))
                            math.ceil(
                                self.ui.FPSTo.value()
                                * (self.amountFrames / self.ui.FPSFrom.value())
                            )
                            self.times = float(self.ui.FPSTo.value()) / float(
                                self.ui.FPSFrom.value()
                            )
                            # log(self.times)

                except Exception as e:
                    pass
            if self.render == "esrgan":
                if self.input_file != "":
                    try:
                        self.resIncrease = int(self.ui.Rife_Times.currentText()[0])
                    except:
                        pass  # hope this works

                    try:
                        try:
                            if self.youtubeFile == True:
                                resolution = self.ytVidRes.replace(
                                    " (Enhanced bitrate)", ""
                                )
                            else:
                                resolution = f"{width}x{height}"
                        except:
                            resolution = f"{width}x{height}"
                        self.ui.FPSPreview.setText(
                            f"RES: {resolution} -> {width*self.resIncrease}x{height*self.resIncrease}"
                        )

                    except Exception as e:
                        tb = traceback.format_exc()
                        log(f"{str(e)} {tb}")
                        # (e,tb)
                        pass
            self.ui.logsPreview.clear()
        except Exception as e:
            tb = traceback.format_exc()
            log(f"{str(e)} {tb}")
            pass

    def reportProgress(self, files_processed):
        files_processed = int(files_processed)

        try:
            if time.time() - self.start_time == 0:
                self.currentRenderFPS = None
            else:
                self.currentRenderFPS = round(
                    (files_processed / (time.time() - self.start_time)), 3
                )

            if self.ncnn:
                # fc is the total file count after interpolation

                if (
                    self.i == 1
                ):  # put every gui change that happens on start of render here
                    if "v4" in self.ui.Rife_Model.currentText().lower():
                        self.times = float(self.ui.FPSTo.value()) / float(
                            self.ui.FPSFrom.value()
                        )
                    fc = int(
                        VideoName.return_video_frame_count(f"{self.input_file}")
                        * self.times
                    )
                    self.filecount = int(fc)
                    total_input_files = fc / self.times
                    total_output_files = fc
                    self.ui.RifePB.setMaximum(total_output_files)

                    self.original_filecount = (
                        self.filecount / self.times
                    )  # this makes the original file count. which is the file count before interpolation

                    self.i = 2
                self.filecount = int(self.original_filecount) * self.times

                ETA = calculateETA(self)
                self.ui.ETAPreview.setText(ETA)

                fp = files_processed
                videos_rendered = -1

                for i in os.listdir(
                    f"{self.settings.RenderDir}/{self.videoName}_temp/output_frames/"
                ):
                    if os.path.isfile(
                        f"{self.settings.RenderDir}/{self.videoName}_temp/output_frames/{i}"
                    ):
                        videos_rendered += 1
                if self.settings.RenderType == "Optimized":
                    try:
                        self.removeLastLineInLogs("Video segments created: ")
                        if videos_rendered == self.interpolation_sessions:
                            self.addLinetoLogs(
                                f"Video segments created: {self.interpolation_sessions}/{self.interpolation_sessions}"
                            )
                        else:
                            self.addLinetoLogs(
                                f"Video segments created: {videos_rendered}/{self.interpolation_sessions}"
                            )
                    except:
                        pass
                self.removeLastLineInLogs("FPS: ")
                self.addLinetoLogs(f"FPS: {self.currentRenderFPS}")

                # Update GUI values
                fp = int(fp)
                self.ui.RifePB.setValue(fp)

                self.i = 2

            if self.cuda:
                if self.i == 1:
                    fc = int(
                        VideoName.return_video_frame_count(f"{self.input_file}")
                        * self.times
                    )
                    self.filecount = fc
                    self.original_filecount = fc / self.times
                    self.filecount = fc
                    self.ui.RifePB.setMaximum(self.filecount)
                try:
                    ETA = calculateETA(self)
                    self.ui.ETAPreview.setText(ETA)
                    self.removeLastLineInLogs("FPS: ")
                    self.addLinetoLogs(f"FPS: {self.currentRenderFPS}")
                except Exception as e:
                    self.ETA = None

                self.i = 2
                self.ui.RifePB.setValue(files_processed)
                self.filecount = int(self.original_filecount) * self.times

            self.ui.processedPreview.setText(
                f"Files Processed: {files_processed} / {int(self.filecount)}"
            )

        except Exception as e:
            print(e)

    def runPB(self):
        self.addLast = False
        self.i = 1
        self.settings = Settings()
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object

        self.worker = workers.pb2X(self.input_file, self.render, self)

        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        self.worker.image_progress.connect(self.imageViewer)
        self.worker.finished.connect(self.endRife)
        # Step 6: Start the thread

        self.thread.start()

        # Final resets

    def imageViewer(self, step):
        if step == "1":
            self.ui.centerLabel.hide()
            self.ui.imageSpacerFrame.hide()
            if "-ncnn-vulkan" in self.AI:
                try:
                    self.pixMap = QPixmap(self.imageDisplay)
                except:
                    pass

        if step == "2":
            try:
                self.pixMap = self.pixMap.scaled(self.width1, self.height1)
                self.ui.imagePreview.setPixmap(self.pixMap)  # sets image preview image
                # self.ui.imagePreview.setMaximumSize(self.width1, self.height1)
                # self.ui.imagePreview.setStyleSheet("background-color: lightblue; border-radius: 100px;")
                # self.round_preview()
            except:
                pass
        if step == "3":
            self.ui.imageSpacerFrame.show()

            self.ui.imagePreview.clear()

    def settings_menu(self):
        item = self.ui.SettingsMenus.currentItem()
        if item.text() == "Video Options":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.show()
            self.ui.GeneralOptionsFrame.hide()
            self.ui.InstallModelsFrame.hide()
            self.fadeIn(self.ui.VideoOptionsFrame)
        if item.text() == "Render Options":
            self.ui.RenderOptionsFrame.show()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.hide()
            self.ui.InstallModelsFrame.hide()
            self.fadeIn(self.ui.RenderOptionsFrame)
        if item.text() == "General":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.show()
            self.ui.InstallModelsFrame.hide()
            self.fadeIn(self.ui.GeneralOptionsFrame)
        if item.text() == "Manage Models":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.hide()
            self.ui.InstallModelsFrame.show()
            self.fadeIn(self.ui.InstallModelsFrame)
            try:
                self.get_models_from_dir("rife")
            except:
                self.ui.defaultRifeModel.hide()
                self.ui.label_13.hide()
                self.ui.label_18.hide()
                self.ui.label_19.hide()

    def setEnsembleMode(self):
        if (
            os.path.exists(
                os.path.join(
                    f"{self.settings.ModelDir}",
                    "rife",
                    f"{self.ui.Rife_Model.currentText()}-ensemble",
                )
            )
            and "rife4" in self.ui.Rife_Model.currentText()
        ):
            self.ui.EnsembleCheckBox.show()
            self.ui.ensembleHelpButton.show()

    def TurnOffCustomFPSMultiplierIfImageExtraction(self):
        if self.ui.ImageExtractionCheckBox.isChecked():
            self.ui.FPSTo.show()
            self.ui.FPSFrom.show()
            self.ui.FPSToSign.show()

        else:
            self.ui.FPSTo.hide()
            self.ui.FPSFrom.hide()
            self.ui.FPSToSign.hide()

    def greyOutRifeTimes(self):
        if (
            "v4" in self.ui.Rife_Model.currentText()
            or "rife4" in self.ui.Rife_Model.currentText()
        ):
            self.ui.Rife_Times.setEnabled(True)

            self.setEnsembleMode()
            self.TurnOffCustomFPSMultiplierIfImageExtraction()

        else:
            self.ui.FPSFrom.hide()
            self.ui.FPSTo.hide()
            self.ui.Rife_Times.setCurrentText("2X")

            self.ui.Rife_Times.setEnabled(False)

            self.ui.EnsembleCheckBox.hide()
            self.ui.ensembleHelpButton.hide()

    def greyOutRealSRTimes(self):
        if self.ui.AICombo.currentText() == "RealESRGAN (NCNN)":
            if (
                self.ui.Rife_Model.currentText() == "Default"
                or self.ui.Rife_Model.currentText() == "General"
            ):
                self.ui.Rife_Times.setCurrentText("4X")
                self.ui.Rife_Times.setEnabled(False)
            elif self.ui.Rife_Model.currentText() == "Animation":
                index = self.ui.Rife_Times.findText("1X")
                if index >= 0:
                    self.ui.Rife_Times.removeItem(index)
                self.ui.Rife_Times.setEnabled(True)
            return
        if (
            self.ui.AICombo.currentText() == "Custom NCNN models"
            or self.ui.AICombo.currentText() == "SPAN (NCNN)"
        ):
            if len(self.ui.Rife_Model.currentText()) > 0:
                self.ui.Rife_Times.clear()
                model = self.ui.Rife_Model.currentText()

                if self.ui.AICombo.currentText() == "Custom NCNN models":
                    modelPath = os.path.join(
                        f"{settings.ModelDir}",
                        "custom_models_ncnn",
                        f"models",
                        f"{self.ui.Rife_Model.currentText()}.param",
                    )

                if self.ui.AICombo.currentText() == "SPAN (NCNN)":
                    modelPath = os.path.join(
                        f"{settings.ModelDir}",
                        "span",
                        f"models",
                        f"{self.ui.Rife_Model.currentText()}.param",
                    )

                try:
                    scale = returnScale(modelPath)
                    self.ui.Rife_Times.addItem(f"{scale}X")
                except Exception as e:
                    log(f"Couldn't Detect Scale! {e}")
                    cantDetectUpscaleTimes(self)
                    self.ui.Rife_Times.addItem("1X")
                    self.ui.Rife_Times.addItem("2X")
                    self.ui.Rife_Times.addItem("3X")
                    self.ui.Rife_Times.addItem("4X")
            self.ui.Rife_Times.setEnabled(True)
        if (
            self.ui.AICombo.currentText() == "Custom CUDA/ROCm models (Nvidia/AMD only)"
            or self.ui.AICombo.currentText() == "Custom TensorRT models"
            or self.ui.AICombo.currentText() == "RealESRGAN TensorRT"
            and len(self.ui.Rife_Model.currentText()) > 0
        ):
            model_path = handleModel(
                "custom-models-cuda", self.ui.Rife_Model.currentText()
            )
            try:
                model = ModelLoader().load_from_file(model_path)
                self.ui.Rife_Times.clear()
                self.ui.Rife_Times.addItem(f"{model.scale}X")
            except Exception as e:
                print(e)
                log(str(f"{e}"))
        if self.ui.AICombo.currentText() == "Waifu2X":
            if self.ui.Rife_Model.currentText() != "cunet":
                self.ui.Rife_Times.setCurrentText("2X")
                self.ui.Rife_Times.setEnabled(False)

            else:
                self.ui.Rife_Times.setEnabled(True)
        if self.ui.AICombo.currentText() == "RealCUGAN":
            self.ui.Rife_Times.setEnabled(True)

    def openFileNameDialog(self, type_of_file, input_file_list):
        files = ""
        for i in input_file_list:
            files += f"*{i} "
        files = files[:-1]
        input_file = QFileDialog.getOpenFileName(
            self,
            "Open File",
            f"{homedir}",
            f"{type_of_file} files ({files});;All files (*.*)",
        )[0]
        if type_of_file == "Video":
            if input_file != "":
                try:
                    self.input_file = input_file
                    cap = cv2.VideoCapture(self.input_file)
                    # Check if the file can be opened successfully
                    if cap.isOpened():
                        cap.release()
                        self.download_youtube_video_command = ""
                        self.localFile = True
                        self.videoName = VideoName.return_video_name(
                            f"{self.input_file}"
                        )
                        if '"' in self.input_file:
                            quotes(self)
                            self.input_file = ""
                        else:
                            self.showChangeInFPS()

                            self.addLinetoLogs(f"Input file = {self.input_file}")
                    else:
                        not_a_video(self)
                except Exception as e:
                    traceback_info = traceback.format_exc()
                    log(f"{e} {traceback_info}")

                    # success!

        if type_of_file == "NCNN Model":
            if (os.path.splitext(input_file)[1])[1:] == "bin":
                if os.path.isfile(input_file):
                    if os.path.basename(input_file) in os.listdir(
                        f"{settings.ModelDir}/custom_models_ncnn/models/"
                    ):
                        alreadyModel(self)
                        return

                    if self.ui.AICombo.currentText() == "Custom NCNN models":
                        CustomModelsNCNN.modelOptions(self)
            else:
                notAModel(self)
                return
            bin = input_file
            inputf = os.path.basename(
                input_file.replace((os.path.splitext(input_file)[1])[1:], "param")
            )
            input_file = QFileDialog.getOpenFileName(
                self, "Open File", f"{homedir}", f".param files ({inputf})"
            )[0]
            if (os.path.splitext(input_file)[1])[1:] == "param":
                if os.path.isfile(input_file):
                    if os.path.basename(input_file) in os.listdir(
                        f"{settings.ModelDir}/custom_models_ncnn/models/"
                    ):
                        alreadyModel(self)
                        return

                    if self.ui.AICombo.currentText() == "Custom NCNN models":
                        CustomModelsNCNN.modelOptions(self)
            else:
                notAModel(self)
                return
            param = input_file
            shutil.copy(
                bin,
                os.path.join(f"{settings.ModelDir}", "custom_models_ncnn", "models"),
            )
            shutil.copy(
                param,
                os.path.join(f"{settings.ModelDir}", "custom_models_ncnn", "models"),
            )
        if type_of_file == "CUDA Model":
            if (
                self.ui.AICombo.currentText()
                == "Custom CUDA/ROCm models (Nvidia/AMD only)"
            ):
                CustomModelsCUDA.modelOptions(self)
            shutil.copy(
                input_file, os.path.join(f"{thisdir}", "models", "custom-models-cuda")
            )

    def openFolderDialog(self):
        output_folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if output_folder != "":
            try:
                if check_for_write_permissions(output_folder):
                    if "/run/user/1000/doc/" in output_folder:
                        output_folder = output_folder.replace("/run/user/1000/doc/", "")
                        output_folder = output_folder.split("/")
                        permissions_dir = ""
                        for index in range(len(output_folder)):
                            if index != 0:
                                permissions_dir += f"{output_folder[index]}/"
                        if homedir not in permissions_dir:
                            output_folder = f"{homedir}/{permissions_dir}"
                        else:
                            output_folder = f"/{permissions_dir}"
                    self.output_folder = output_folder
                else:
                    no_perms_change_setting(self)
            except:
                self.showDialogBox("Invalid Directory")

    def ignore_vram_popup(self):
        settings.change_setting("ignoreVramPopup", "True")

    def showPopup(self, message, ignore_function):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("REAL Video Enhancer")
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Ignore)

        # Connect the button signals to functions
        ignore_button = msg_box.button(QMessageBox.Ignore)
        ignore_button.clicked.connect(ignore_function)

        result = msg_box.exec_()

    def ignoreButtonClicked(self):
        log("Ignore button clicked!")

    def cuganDenoiseLevel(self):
        try:
            if self.ui.AICombo.currentText() == "RealCUGAN":
                if int(self.ui.Rife_Times.currentText()[0]) > 2:
                    self.ui.denoiseLevelSpinBox.setValue(0)
                    self.ui.denoiseLevelSpinBox.setSingleStep(3)
                else:
                    self.ui.denoiseLevelSpinBox.setSingleStep(1)
        except:
            pass

    def incrementcuganDenoiseLevel(self):
        if self.ui.AICombo.currentText() == "RealCUGAN":
            if int(self.ui.Rife_Times.currentText()[0]) > 2:
                if (
                    self.ui.denoiseLevelSpinBox.value() == 1
                    or self.ui.denoiseLevelSpinBox.value() == 2
                ):
                    self.ui.denoiseLevelSpinBox.setValue(3)

    def setDisableEnable(self, mode):
        self.ui.FPSTo.setDisabled(mode)
        self.ui.AICombo.setDisabled(mode)
        self.ui.EnsembleCheckBox.setDisabled(mode)
        self.ui.ensembleHelpButton.setDisabled(mode)
        self.ui.RifeStart.setDisabled(mode)
        self.ui.Input_video_rife.setDisabled(mode)
        self.ui.Input_video_rife_url.setDisabled(mode)
        self.ui.Output_folder_rife.setDisabled(mode)
        self.ui.Rife_Model.setDisabled(mode)
        self.ui.Rife_Times.setDisabled(mode)
        try:
            if (
                self.AI == "rife-ncnn-vulkan"
                and not "v4" in self.ui.Rife_Model.currentText().lower()
            ):
                self.ui.Rife_Times.setDisabled(True)
        except:
            pass
        self.ui.verticalTabWidget.tabBar().setDisabled(mode)
        self.ui.denoiseLevelSpinBox.setDisabled(mode)
        self.ui.InstallModelsFrame.setDisabled(mode)
        self.ui.SettingsMenus.setDisabled(mode)
        self.ui.modeCombo.setDisabled(mode)
        self.ui.ImageExtractionCheckBox.setDisabled(mode)

    def endRife(
        self,
    ):  # Crashes most likely due to the fact that it is being ran in a different thread
        if "cuda" in self.AI or "ncnn-python" in self.AI:
            while self.CudaRenderFinished == False:
                sleep(1)
        self.file_drop_widget.show()
        self.ui.QueueListWidget.hide()
        try:
            self.RPC.clear(pid=os.getpid())
        except:
            pass
        self.ui.RifePause.hide()
        self.ui.RifeResume.hide()
        self.ui.QueueButton.hide()
        self.ui.centerLabel.show()

        self.addLinetoLogs(f"Finished! Output video: {self.output_file}")
        self.setDisableEnable(False)
        self.ui.RifePB.setValue(self.ui.RifePB.maximum())
        self.ui.ETAPreview.setText("ETA: 00:00:00")
        self.ui.imagePreview.clear()

        self.ui.imageSpacerFrame.show()

        remaining_time = int(time.time() - self.start_time)

        hours, minutes, seconds = convertTime(remaining_time)

        self.addLinetoLogs(f"Total Time: {hours}:{minutes}:{seconds}")

    def showDialogBox(self, message, displayInfoIcon=False):
        icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        if displayInfoIcon == True:
            msg.setIconPixmap(icon.pixmap(32, 32))
        msg.setText(f"{message}")
        msg.exec_()

    def showQuestionBox(self, message):
        reply = QMessageBox.question(
            self, "", f"{message}", QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            return True
        else:
            return False

    def closeEvent(self, event):
        if self.input_file != "":
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "Are you sure you want to exit?\n(The current render will be killed)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

        else:
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

        if reply == QMessageBox.Yes:
            self.on = False
            event.accept()
            if self.input_file != "":
                try:
                    shutil.rmtree(
                        os.path.join(settings.RenderDir, f"{self.videoName}_temp")
                    )
                    shutil.rmtree(os.path.join(thisdir, self.videoName))

                except Exception as e:
                    log(str(e))
                try:
                    self.ffmpeg.terminate()
                except:
                    pass
                try:
                    self.renderAI.terminate()
                except:
                    pass

                try:
                    shutil.rmtree(
                        os.path.join(settings.RenderDir, f"{self.videoName}_temp")
                    )
                except:
                    pass
                exit()
        else:
            event.ignore()

    def addLinetoLogs(self, line, remove_text=""):
        if line != "REMOVE_LAST_LINE" or remove_text != "":
            self.ui.logsPreview.append(f"{line}")
        else:
            self.removeLastLineInLogs(remove_text)

    def update_last_line(self, new_line_text):
        # Assuming line number is 2 (index 1) - replace with the desired line number
        line_number = 1

        cursor = self.ui.logsPreview.textCursor()
        cursor.movePosition(cursor.Start)
        for _ in range(line_number):
            cursor.movePosition(cursor.Down, cursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(new_line_text)

    def removeLastLineInLogs(
        self, text_in_line
    ):  # takes in text in line and removes every line that has that specific text.
        text = self.ui.logsPreview.toPlainText().split("\n")
        text1 = []
        for i in text:
            if i != " " or i != "":
                if len(i) > 3:
                    text1.append(i)
        text = text1
        display_text = ""
        for i in text:
            if text_in_line not in i:
                display_text += f"{i}"
                if i != " ":
                    display_text += "\n"
            else:
                pass
        self.ui.logsPreview.clear()

        self.ui.logsPreview.setText(display_text)
        scroll_bar = self.ui.logsPreview.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())


try:
    if os.path.isfile(f"{thisdir}/files/settings.txt") == False:
        ManageFiles.create_folder(f"{thisdir}/files")
        ManageFiles.create_file(f"{thisdir}/files/settings.txt")
    settings = Settings()
except Exception as e:
    traceback_info = traceback.format_exc()
    log(f"{e} {traceback_info}")


def excepthook(type, value, extraceback):
    # Extract the filename and line number where the exception occurred
    filename = extraceback.tb_frame.f_code.co_filename
    lineno = extraceback.tb_lineno

    # Log the exception details
    log(f"Exception Type: {type}")
    log(f"Exception Value: {value}")
    log(f"Exception occurred in file: {filename}, line {lineno}")

    # Print the traceback
    log("Traceback:")
    traceback.print_tb(extraceback)

    # Format the traceback as a string
    tb_str = traceback.format_exc()

    # Log the formatted traceback
    log(f"Traceback (formatted):\n{tb_str}")

    # Display an error message to the user
    error_message = (
        f"An unhandled exception occurred: {value} (in {filename}, line {lineno})"
    )
    log(f"ERROR: {error_message}")
    QMessageBox.critical(None, "Error", error_message, QMessageBox.Ok)


import functools
import inspect


def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        caller_frame = inspect.currentframe().f_back
        caller_file = inspect.getfile(caller_frame)
        caller_function = inspect.getframeinfo(caller_frame).function
        tb = traceback.format_exc()
        print(
            f"Function {func.__name__} called from {caller_function} in file {caller_file} {tb}"
        )
        return func(*args, **kwargs)

    return wrapper


# Applying the decorator to all functions in a script
def apply_decorator_to_all_functions(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            setattr(module, name, log_function_call(obj))


# Applying the decorator to all functions in the current module
apply_decorator_to_all_functions(globals())


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor

# Force the style to be the same on all OSs:
theme.set_theme(app)
log("Program Started")
sys.excepthook = excepthook
app.exec_()
log("exit")

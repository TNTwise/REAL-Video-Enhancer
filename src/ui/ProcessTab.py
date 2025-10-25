import subprocess
import os
from threading import Thread
import re
import math
from multiprocessing import shared_memory

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor, QFontMetrics
from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtWidgets import QMessageBox

from .RenderQueue import RenderQueue

from .AnimationHandler import AnimationHandler
from .QTcustom import (
    RegularQTPopup,
    UpdateGUIThread,
    show_layout_widgets,
    hide_layout_widgets,
)
from ..constants import (
    BACKEND_PATH,
    PYTHON_EXECUTABLE_PATH,
    MODELS_PATH,
    CUSTOM_MODELS_PATH,
    IMAGE_SHARED_MEMORY_ID,
    PAUSED_STATE_SHARED_MEMORY_ID,
    INPUT_TEXT_FILE,
    FFMPEG_PATH,
    CWD,
    PLATFORM,
)
from ..Util import (
    log,
)
from ..DownloadModels import DownloadModel
from .SettingsTab import Settings
from ..DiscordRPC import DiscordRPC
from ..ModelHandler import getModels
from .RenderQueue import RenderOptions


class ProcessTab:
    def __init__(self, parent, settings: Settings):
        self.parent = parent
        self.renderTextOutputList = None
        self.isOverwrite = False
        self.outputVideoHeight = None
        self.outputVideoWidth = None
        self.currentFrame = 0
        self.fps = 0
        self.eta = 0
        self.isPreview = False
        self.userKilled = False
        self.currentRenderOptions = None
        self.status = "Idle"
        self.animationHandler = AnimationHandler()
        self.tileUpAnimationHandler = AnimationHandler()
        self.tileDownAnimationHandler = AnimationHandler()
        self.settings = settings
        self.return_codes = []

        self.qualityToCRF = {
            "Low": "28",
            "Medium": "23",
            "High": "18",
            "Very High": "15",
        }
        # encoder dict
        # key is the name in RVE gui
        # value is the encoder used

        # get default backend
        self.QConnect()
        self.populateModels(self.parent.backendComboBox.currentText())

    def populateModels(self, backend) -> dict:
        """
        returns
        the current models available given a method (interpolate, upscale) and a backend (ncnn, tensorrt, pytorch)
        """
        interpolateModels, upscaleModels, deblurModels, denoiseModels, decompressModels = getModels(backend)
        self.parent.interpolateModelComboBox.clear()
        self.parent.upscaleModelComboBox.clear()
        self.parent.deblurModelComboBox.clear()
        self.parent.denoiseModelComboBox.clear()
        self.parent.decompressModelComboBox.clear()
        self.parent.interpolateModelComboBox.addItems(
            list(interpolateModels.keys())
        )
        self.parent.interpolateModelComboBox.setCurrentIndex(len(list(interpolateModels.keys()))-1)
        self.parent.upscaleModelComboBox.addItems(list(upscaleModels.keys()))
        self.parent.deblurModelComboBox.addItems(list(deblurModels.keys()))
        self.parent.denoiseModelComboBox.addItems(list(denoiseModels.keys()))
        self.parent.decompressModelComboBox.addItems(list(decompressModels.keys()))
    def onTilingSwitch(self):
        if self.parent.tilingCheckBox.isChecked():
            self.parent.tileSizeContainer.setVisible(True)
            self.tileDownAnimationHandler.dropDownAnimation(
                self.parent.tileSizeContainer
            )
        else:
            self.tileUpAnimationHandler.moveUpAnimation(self.parent.tileSizeContainer)
            self.parent.tileSizeContainer.setVisible(False)

    def openOutputFolderInExplorer(self):
        output_folder = os.path.dirname(self.parent.outputFileText.text())
        if os.path.isdir(output_folder):
            if PLATFORM == "win32":
                os.startfile(output_folder)
            elif PLATFORM == "darwin":
                subprocess.Popen(["open", output_folder])
            else:
                subprocess.Popen(["xdg-open", output_folder])

    def QConnect(self):
        # connect file select buttons
        self.parent.addToRenderQueueButton.clicked.connect(self.parent.addToRenderQueue)
        self.parent.RemoveFromRenderQueue.clicked.connect(
            self.parent.renderQueue.remove
        )
        self.parent.MoveUpRenderQueue.clicked.connect(
            lambda: self.parent.renderQueue.moveitem("up")
        )
        self.parent.MoveDownRenderQueue.clicked.connect(
            lambda: self.parent.renderQueue.moveitem("down")
        )
        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.batchSelectButton.clicked.connect(self.parent.openBatchFiles)
        self.parent.inputFileText.textChanged.connect(self.parent.loadVideo)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        self.parent.openOutputFolderButton.clicked.connect(self.openOutputFolderInExplorer)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        # set tile size visible to false by default
        self.parent.tileSizeContainer.setVisible(False)
        # set slo mo container visable to false by default
        
        self.parent.interpolateContainer_2.setVisible(False)
        # connect up tilesize container visiable
        self.parent.tilingCheckBox.stateChanged.connect(self.onTilingSwitch)

        self.parent.interpolationMultiplierSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )

        self.parent.upscaleModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.upscaleScaleSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.interpolateModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.decompressModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.interpolateCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.upscaleCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.deblurCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.denoiseCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.decompressCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)   

        self.parent.backendComboBox.currentIndexChanged.connect(
            lambda: self.populateModels(self.parent.backendComboBox.currentText())
        )
        self.parent.EncoderCommand.textChanged.connect(lambda: self.parent.EncoderCommand.setFixedWidth(max(50, QFontMetrics(self.parent.EncoderCommand.font()).horizontalAdvance(self.parent.EncoderCommand.text()) + 10)))
        # connect up pausing
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.pauseRenderButton.clicked.connect(self.pauseRender)
        self.parent.killRenderButton.clicked.connect(self.killRenderProcess)

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.userKilled = True
            self.renderProcess.terminate()
        except AttributeError:
            log("No render process!")

    def pauseRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 1  # 1 = True
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)

    def resumeRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 0  # 0 = False
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.onRenderButtonsContiainer.setEnabled(True)
        self.parent.startRenderButton.setVisible(False)

    def startGUIUpdate(self):
        self.workerThread = UpdateGUIThread(
            parent=self,
            imagePreviewSharedMemoryID=IMAGE_SHARED_MEMORY_ID,
        )
        self.workerThread.latestPreviewPixmap.connect(self.updateProcessTab)
        self.workerThread.finished.connect(self.guiChangesOnRenderCompletion)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def splitListIntoStringWithNewLines(self, string_list: list[str]):
        # Join the strings with newline characters
        return "\n".join(string_list)
        # Set the text to the QTextEdit

    def questionToOverride(self):
        reply = QMessageBox.question(
            self.parent,
            "",
            "Output files in render queue already exist, do you want to overwrite?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        return reply == QMessageBox.Yes  # type: ignore

    def askForOverwrite(self, outputPath):
        if os.path.isfile(outputPath):
            self.isOverwrite = self.questionToOverride()
            if not self.isOverwrite:
                self.onRenderCompletion()
                self.guiChangesOnRenderCompletion()
                return False
        return True

    def checkForOverwrite(self, renderQueue: RenderQueue):
        for renderOptions in renderQueue.getQueue():
            if not self.askForOverwrite(renderOptions.outputPath):
                return False
        return True

    def run(
        self,
        renderQueue: RenderQueue,
    ):
        self.return_codes = [] # reset return codes
        self.userKilled = False # reset userkilled
        # gui changes
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(False)
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.resumeRender)

        if not self.checkForOverwrite(renderQueue):
            return

        self.startDiscordRPC()
        self.settings.readSettings()

        self.startGUIUpdate()
        writeThread = Thread(target=lambda: self.renderToPipeThread(renderQueue))
        writeThread.start()

    def startDiscordRPC(self):
        if self.settings.settings["discord_rich_presence"] == "True":
            try:
                self.discordRPC = DiscordRPC()
                self.discordRPC.start_discordRPC()
            except Exception:
                pass

    def createPausedSharedMemory(self):
        try:
            self.pausedSharedMemory = shared_memory.SharedMemory(
                name=PAUSED_STATE_SHARED_MEMORY_ID, create=True, size=1
            )
        except FileExistsError:
            log("FileExistsError! Using existing paused shared memory")
            self.pausedSharedMemory = shared_memory.SharedMemory(
                name=PAUSED_STATE_SHARED_MEMORY_ID
            )

    def renderToPipeThread(
        self,
        renderQueue: RenderQueue,
    ):
        self.createPausedSharedMemory()

        for renderOptions in renderQueue.getQueue():

            self.isPreview = renderOptions.isPreview
            self.currentRenderOptions = renderOptions

            self.workerThread.setOutputVideoRes(
                renderOptions.videoWidth * renderOptions.overrideUpscaleScale,
                renderOptions.videoHeight * renderOptions.overrideUpscaleScale,
            )
            self.parent.progressBar.setRange(
                0,
                # only set the range to multiply the frame count if the method is interpolate
                int(
                    renderOptions.videoFrameCount
                    * math.ceil(renderOptions.interpolateTimes)
                ),
            )
            command = self.build_command(renderOptions)
            log(str(command))

            kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "universal_newlines": True,

            }

            if PLATFORM == "win32":
                kwargs["startupinfo"] = subprocess.STARTUPINFO()
                kwargs["startupinfo"].dwFlags |= subprocess.STARTF_USESHOWWINDOW


            self.renderProcess = subprocess.Popen(
                command,
                **kwargs,
            )
            textOutput = []
            for line in iter(self.renderProcess.stdout.readline, b""):
                if self.renderProcess.poll() is not None:
                    break  # Exit the loop if the process has terminated

                try:
                    line = str(line.strip())
                    if "it/s" in line:
                        textOutput = textOutput[:-1]
                    if "FPS" in line:
                        textOutput = textOutput[
                            :-1
                        ]  # slice the list to only get the last updated data
                        self.currentFrame = int(
                            re.search(r"Current Frame: (\d+)", line).group(1)
                        )
                        self.fps = re.search(r"FPS: (\d+)", line).group(1)
                        self.eta = re.search(r"ETA: (.+)", line).group(1)
                        self.status = "Rendering"

                    if "this may take a while" in line.lower():
                        self.status = "Building Engine"


                    if any(char.isalpha() for char in line):
                        textOutput.append(line)
                    # self.setRenderOutputContent(textOutput)
                    self.renderTextOutputList = textOutput.copy()
                except Exception as e:
                    pass
                if "Time to complete render" in line:
                    break
            self.return_codes.append(self.renderProcess.wait())
            for line in textOutput:
                if len(line) > 2:
                    log(line)

            self.parent.OutputFilesListWidget.addItem(
                renderOptions.outputPath
            )  # add the file to the list widget

            self.workerThread.unlink_shared_memory()
        try:
            self.pausedSharedMemory.close()
            self.pausedSharedMemory.unlink()
        except Exception: # too lazy to patch why this errors maybe on exit
            pass

        renderQueue.clear()
        self.onRenderCompletion()

    def guiChangesOnRenderCompletion(self):
        if all(return_code == 0 or return_code == 3221225477 for return_code in self.return_codes) or self.userKilled: # 3221225477 comes up when using ncnn on windows for some reason, but no error in output itself.
            log("All render processes completed successfully")
        else:
            log("Some render processes failed: Error code: " + str(self.return_codes))
            RegularQTPopup("Rendering failed! Please check the logs tab!")
        # Have to swap the visibility of these here otherwise crash for some reason
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setEnabled(True)
        self.parent.outputFileText.setEnabled(True)
        self.parent.inputFileText.setEnabled(True)
        self.parent.previewLabel.clear()
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        self.parent.enableProcessPage()
        self.parent.startRenderButton.setVisible(True)
        self.parent.FPS.setText("FPS: ")
        self.parent.ETA.setText("ETA: ")
        self.parent.STATUS.setText("Status: ")
        if self.currentRenderOptions.isPreview:
            from PySide6.QtMultimedia import QMediaPlayer
            try:
                def onScroll(preview:QMediaPlayer, value):
                    preview.setPosition(value)

                self.parent.renderQueue.clear()

                player = QMediaPlayer()
                player.setSource(QUrl.fromLocalFile(self.currentRenderOptions.outputPath))
                player.setVideoOutput(self.parent.VideoPreview)
                player.play()
                player.pause()
                player.setPosition(5)
                self.parent.VideoPreview.show()
                self.parent.VideoPreview.setVisible(True)
                self.parent.previewLabel.setVisible(False)
                self.parent.timeInVideoScrollBar.setRange(0, (((self.currentRenderOptions.endTime-self.currentRenderOptions.startTime))*10)-1) # convert to ms
                self.parent.timeInVideoScrollBar.valueChanged.connect(lambda: onScroll(player, int(self.parent.timeInVideoScrollBar.value()*100)))

            except Exception as e:
                log(f"Error: {e}")


    def onRenderCompletion(self):
        try:
            self.renderProcess.wait()
        except Exception:
            pass
        # Have to swap the visibility of these here otherwise crash for some reason
        if (
            self.settings.settings["discord_rich_presence"] == "True"
        ):  # only close if it exists
            self.discordRPC.closeRPC()
        try:
            self.workerThread.stop()
            self.workerThread.quit()
            self.workerThread.wait()
        except Exception:
            pass  # pass just incase internet error caused a skip

    def getRoundedPixmap(self, pixmap, corner_radius):
        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)  # type: ignore

        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, size.width(), size.height(), corner_radius, corner_radius
        )

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        rounded_pixmap = QPixmap(size)
        rounded_pixmap.fill(Qt.transparent)  # type: ignore

        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore
        painter.drawPixmap(0, 0, mask)
        painter.end()

        return rounded_pixmap

    def modelNameToFile(self):
        pass

    def updateProcessTab(self, qimage: QtGui.QImage):
        """
        Called by the worker QThread, and updates the GUI elements: Progressbar, Preview, FPS
        """

        if self.renderTextOutputList is not None:
            # print(self.renderTextOutputList)
            self.parent.renderOutput.setPlainText(
                self.splitListIntoStringWithNewLines(self.renderTextOutputList)
            )
            scrollbar = self.parent.renderOutput.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            self.parent.progressBar.setValue(self.currentFrame)
            self.parent.FPS.setText(f"FPS: {self.fps}")
            self.parent.ETA.setText(f"ETA: {self.eta}")
            self.parent.STATUS.setText(f"Status: {self.status}")
        if not qimage.isNull():
            label_width = self.parent.previewLabel.width()
            label_height = self.parent.previewLabel.height()

            p = qimage.scaled(
                label_width, label_height, Qt.AspectRatioMode.KeepAspectRatio
            )  # type: ignore
            pixmap = QtGui.QPixmap.fromImage(p)

            roundedPixmap = self.getRoundedPixmap(pixmap, corner_radius=10)
            self.parent.previewLabel.setPixmap(roundedPixmap)

    def build_command(self, renderOptions: RenderOptions):
        if (
            renderOptions.backend == "pytorch (cuda)"
            or renderOptions.backend == "pytorch (rocm)"
            or renderOptions.backend == "pytorch (xpu)"
            or renderOptions.backend == "pytorch (mps)"
        ):
            renderOptions.backend = (
                "pytorch"  # pytorch is the same for both cuda and rocm
            )

        command = [
            f"{PYTHON_EXECUTABLE_PATH}",
            "-W",
            "ignore",
            os.path.join(BACKEND_PATH, "rve-backend.py"),
            "-i",
            renderOptions.inputFile,
            "-o",
            renderOptions.outputPath,
            "-b",
            f"{renderOptions.backend}",
            "--precision",
            f"{self.settings.settings['precision']}",
            "--custom_encoder",
            f" {renderOptions.encoderCommand} ",
            "--tensorrt_opt_profile",
            f"{self.settings.settings['tensorrt_optimization_level']}",
            "--pause_shared_memory_id",
            f"{PAUSED_STATE_SHARED_MEMORY_ID}",
            "--ncnn_gpu_id",
            f"{self.settings.settings['ncnn_gpu_id']}",
            "--pytorch_gpu_id",
            f"{self.settings.settings['pytorch_gpu_id']}",
            "--cwd",
            f"{CWD}",

        ]

        if renderOptions.upscaleModelFile:
            modelPath = os.path.join(MODELS_PATH, renderOptions.upscaleModelFile)
            if renderOptions.upscaleModelArch == "custom":
                modelPath = os.path.join(
                    CUSTOM_MODELS_PATH, renderOptions.upscaleModelFile
                )
            command += [
                "--upscale_model",
                modelPath,
            ]
            command += ["--override_upscale_scale", f"{renderOptions.overrideUpscaleScale}"]
            
        if renderOptions.tilingEnabled:
            command += [
                "--tilesize",
                f"{renderOptions.tilesize}",
            ]
            

        if renderOptions.interpolateModelFile:
            command += [
                "--interpolate_model",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.interpolateModelFile,
                ),
                "--interpolate_factor",
                f"{renderOptions.interpolateTimes}",
            ]
            if renderOptions.sloMoMode:
                command += [
                    "--slomo_mode",
                ]
            if renderOptions.dyanmicScaleOpticalFlow:
                command += [
                    "--dynamic_scaled_optical_flow",
                ]
            if renderOptions.ensemble:
                command += [
                    "--ensemble",
                ]
        
        if renderOptions.deblurModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.deblurModelFile,
                ),
            ]
        if renderOptions.denoiseModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.denoiseModelFile,
                ),
            ]
        if renderOptions.decompressModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.decompressModelFile,
                ),
            ]

        if self.settings.settings["auto_border_cropping"] == "True":
            command += [
                "--border_detect",
            ]

        if self.settings.settings["dynamic_tensorrt_engine"] == "True":
            command += [
                "--tensorrt_dynamic_shapes",
            ]

        if self.settings.settings["preview_enabled"] == "True":
            command += [
                "--preview_shared_memory_id",
                f"{IMAGE_SHARED_MEMORY_ID}",
            ]

        if self.settings.settings["scene_change_detection_enabled"] == "False":
            command += ["--scene_detect_method", "none"]
        else:
            command += [
                "--scene_detect_method",
                self.settings.settings["scene_change_detection_method"],
                "--scene_detect_threshold",
                self.settings.settings["scene_change_detection_threshold"],
            ]

        if renderOptions.benchmarkMode:
            command += ["--benchmark"]

        if self.settings.settings["uhd_mode"] == "True":
            if renderOptions.videoWidth > 1920 or renderOptions.videoHeight > 1080:
                command += ["--UHD_mode"]
                log("UHD mode enabled")

        if self.isOverwrite:
            command += ["--overwrite"]

        if renderOptions.hdrMode:
            command += ["--hdr_mode"]



        return command

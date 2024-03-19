import src.programData.return_data as return_data
import os
from src.programData.settings import *
import src.runAI.transition_detection
from src.programData.return_data import *
from src.misc.messages import *
from src.runAI.discord_rpc import *
import os
from modules.commands import *
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
)
import modules.upscale as upscale
from src.misc.log import *

# this file changes the GUI aspects of the AI
thisdir = src.programData.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
from PyQt5.QtCore import QThread
import src.runAI.workers as workers
import src.programData.checks as checks


def initializeUpscale(
    self, AI
):  # 1st stage in preparing render, starts all worker threads
    try:
        if self.input_file != "":
            settings = Settings()
            os.system(f'rm -rf "{settings.RenderDir}/{self.videoName}_temp/"')
            self.ui.logsPreview.clear()

            self.render = "esrgan"
            self.AI = AI

            self.setDisableEnable(True)

            if settings.DiscordRPC == "Enabled":
                try:
                    start_discordRPC(self, "Upscaling")
                except:
                    print("No discord on this machine")

            realESRGAN_Model = self.ui.Rife_Model.currentText()
            realESRGAN_Times = self.ui.Rife_Times.currentText()[0]
            if AI == "realesrgan-ncnn-vulkan" or "custom-models-ncnn-vulkan":
                if self.ui.gpuIDSpinBox.value() == -1:
                    self.showDialogBox(
                        "ESRGAN does not support CPU inference, using GPU inference instead."
                    )
                if realESRGAN_Model == "Default":
                    self.realESRGAN_Model = "-n realesrgan-x4plus -s 4"

                if realESRGAN_Model == "Animation":
                    self.realESRGAN_Model = (
                        f"-n realesr-animevideov3 -s {realESRGAN_Times}"
                    )
                else:
                    self.realESRGAN_Model = (
                        f"-n {self.ui.Rife_Model.currentText()} -s {realESRGAN_Times}"
                    )

            self.ui.logsPreview.append(f"[Extracting Frames]")
            self.ui.ETAPreview.setText("ETA:")
            self.ui.processedPreview.setText("Files Processed:")

            self.upscaleThread = QThread()
            # Step 3: Create a worker object

            self.upscaleWorker = workers.upscale(self)

            # Step 4: Move worker to the thread
            self.upscaleWorker.moveToThread(self.upscaleThread)
            # Step 5: Connect signals and slots
            self.upscaleThread.started.connect(self.upscaleWorker.finishRenderSetup)
            self.upscaleWorker.finished.connect(self.upscaleThread.quit)
            self.upscaleWorker.finished.connect(self.upscaleWorker.deleteLater)
            self.upscaleThread.finished.connect(self.upscaleThread.deleteLater)
            self.upscaleWorker.log.connect(self.addLinetoLogs)
            self.upscaleWorker.removelog.connect(self.removeLastLineInLogs)
            # Step 6: Start the thread

            self.upscaleThread.start()
            self.runPB()
        else:
            self.showDialogBox(no_input_file)
    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f"ERROR: {e} {traceback_info}")
        self.showDialogBox(e)


def start_upscale(self, AI):  # command linked directly to upscale buttons
    try:
        if self.input_file != "":
            self.render = "esrgan"
            has_enough_space, predicted_space, total_space = (
                checks.check_if_enough_space(self.input_file, self.render, self.times)
            )
            if self.input_file.count("'") > 0 or '"' in self.input_file:
                quotes(self)
                return
            has_enough_output_space, predicted_output_space, total_output_space = (
                checks.check_if_enough_space_output_disk(
                    self.input_file, self.render, self.times
                )
            )
            if not has_enough_output_space:
                if not_enough_output_storage(
                    self, predicted_output_space, total_output_space
                ):
                    initializeUpscale(self, AI)
                else:
                    pass
            if has_enough_space:
                initializeUpscale(self, AI)
            elif not_enough_storage(self, predicted_space, total_space):
                initializeUpscale(self, AI)
            else:
                pass

        else:
            no_input_file(self)

    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f"ERROR: {e} {traceback_info}")
        self.showDialogBox(e)

import time
from multiprocessing import shared_memory

if __name__ != "__main__":
    from .utils.LogConfig import get_logger
    from .utils.PauseManager import PauseManager
    from .utils.RealTimePrint import RealTimePrint
    from .utils.Util import padFrame

    logger = get_logger(__name__)
else:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def convertTime(remaining_time):
    """
    Converts seconds to hours, minutes and seconds
    """
    hours = remaining_time // 3600
    remaining_time -= 3600 * hours
    minutes = remaining_time // 60
    remaining_time -= minutes * 60
    seconds = remaining_time
    if minutes < 10:
        minutes = str(f"0{minutes}")
    if seconds < 10:
        seconds = str(f"0{seconds}")
    return hours, minutes, seconds


class InformationWriteOut:
    def __init__(
        self,
        sharedMemoryID,  # image memory id
        sharedMemoryChunkSize,  # size of the image memory
        paused_shared_memory_id,
        outputWidth,
        outputHeight,
        croppedOutputWidth,
        croppedOutputHeight,
        totalOutputFrames,
        border_detect: bool = False,
        hdr_mode: bool = False,
    ):
        self.startTime = time.time()
        self.sharedMemoryID = sharedMemoryID
        self.paused_shared_memory_id = paused_shared_memory_id
        self.width = outputWidth
        self.height = outputHeight
        self.croppedOutputWidth = croppedOutputWidth
        self.croppedOututHeight = croppedOutputHeight
        self.totalOutputFrames = totalOutputFrames
        self.border_detect = border_detect
        self.previewFrame = None
        self.last_length = 0
        self.framesRendered = 1
        self.total_paused_time_seconds = 0
        self.hdr_mode = hdr_mode
        self.sharedMemoryChunkSize = sharedMemoryChunkSize

        if self.sharedMemoryID is not None:
            while True:
                try:
                    self.shm = shared_memory.SharedMemory(name=self.sharedMemoryID)
                    logger.info("Connected to shared memory: %s", self.sharedMemoryID)
                    break
                except FileNotFoundError:
                    logger.info(
                        "Waiting for shared memory to be created: %s",
                        self.sharedMemoryID,
                    )
                    time.sleep(0.5)

        self.pausedManager = PauseManager(paused_shared_memory_id)
        self.realTimePrint = RealTimePrint()
        self.isPaused = False
        self.stop = False

    def get_is_paused(self):
        return self.isPaused

    def calculateETA(self, framesRendered):
        """
        Calculates ETA

        Gets the time for every frame rendered by taking the
        elapsed time / completed iterations (files)
        remaining time = remaining iterations (files) * time per iteration

        """
        # Estimate the remaining time
        elapsed_time = time.time() - self.startTime
        time_per_iteration = elapsed_time / framesRendered
        remaining_iterations = self.totalOutputFrames - framesRendered
        remaining_time = remaining_iterations * time_per_iteration
        remaining_time = int(remaining_time)
        # convert to hours, minutes, and seconds
        hours, minutes, seconds = convertTime(remaining_time)
        return f"{hours}:{minutes}:{seconds}"

    def update(self, preview_frame):
        self.previewFrame = preview_frame
        self.framesRendered += 1

    def stopWriting(self):
        self.stop = True

    def writeOutInformation(self):
        """
        Fcs = framechunksize
        """
        while (not self.stop) and self.framesRendered > 0:
            time.sleep(
                0.5
            )  # setting this to a higher value will reduce the cpu usage, and increase fps
            self.isPaused = self.pausedManager.pause_manager()

            if self.isPaused:
                pause_start_time = time.time()
                while self.isPaused and not self.stop:
                    self.isPaused = self.pausedManager.pause_manager()
                    time.sleep(0.5)
                pause_end_time = time.time()
                paused_duration = pause_end_time - pause_start_time
                self.total_paused_time_seconds += paused_duration

            # print out data to stdout
            fps = round(
                self.framesRendered
                / (time.time() - self.startTime - self.total_paused_time_seconds)
            )
            eta = self.calculateETA(framesRendered=self.framesRendered)
            message = f"FPS: {fps} Current Frame: {self.framesRendered} ETA: {eta}"
            self.realTimePrint.realTimePrint(message)

            if self.sharedMemoryID is not None and self.previewFrame is not None:
                # Update the shared array
                padded_frame = padFrame(  # pad frame in case of border detect
                    self.previewFrame,
                    self.width,
                    self.height,
                    self.croppedOutputWidth,
                    self.croppedOututHeight,
                )
                self.shm.buf[: self.sharedMemoryChunkSize] = bytes(padded_frame)

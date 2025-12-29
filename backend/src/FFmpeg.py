from multiprocessing import shared_memory
import sys
import time
import numpy as np

if __name__ != "__main__":
    from .utils.Util import log, padFrame, subprocess_popen_without_terminal
else:
    def log(message):
        print(message)

def hdr_to_sdr(hdr_frame, width, height):
    """
    Converts HDR frame (uint16) to SDR (uint8) using tone mapping
    """
    # Convert buffer to numpy array
    hdr_frame = np.frombuffer(hdr_frame, dtype=np.uint16)
    hdr_frame = hdr_frame.reshape((height, width, 3))
    
    # Normalize to 0-1 range
    hdr_normalized = hdr_frame.astype(np.float32) / 65535.0
    
    # Apply tone mapping (Reinhard operator)
    # This preserves details in highlights and shadows
    L = 0.2126 * hdr_normalized[:,:,0] + 0.7152 * hdr_normalized[:,:,1] + 0.0722 * hdr_normalized[:,:,2]
    L_white = np.max(L)  # Max luminance value
    
    L_tone_mapped = L * (1 + L / (L_white * L_white)) / (1 + L)
    
    # Scale each color channel
    ratio = np.divide(L_tone_mapped, L, out=np.ones_like(L), where=L!=0)
    ratio = ratio[:,:,np.newaxis]
    sdr_normalized = hdr_normalized * ratio
    
    # Convert back to uint8 (0-255)
    sdr_frame = (sdr_normalized * 255).clip(0, 255).astype(np.uint8)
    
    return sdr_frame.tobytes()


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


class PauseManager:
    def __init__(self, paused_shared_memory_id):
        self.isPaused = False
        self.prevState = None
        self.paused_shared_memory_id = paused_shared_memory_id
        if self.paused_shared_memory_id is not None:
            try:
                self.pausedSharedMemory = shared_memory.SharedMemory(
                    name=self.paused_shared_memory_id
                )
            except FileNotFoundError:
                log(f"FileNotFoundError! Creating new paused shared memory: {self.paused_shared_memory_id}")
                self.pausedSharedMemory = shared_memory.SharedMemory(
                    name=self.paused_shared_memory_id, create=True, size=1
                )
    def pause_manager(self):
        if self.paused_shared_memory_id is not None:
            return self.pausedSharedMemory.buf[0] == 1

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
        self.hdr_mode = hdr_mode
        self.sharedMemoryChunkSize = sharedMemoryChunkSize

        if self.sharedMemoryID is not None:
            while True:
                try:
                    self.shm = shared_memory.SharedMemory(
                        name=self.sharedMemoryID
                    )
                    break
                except FileNotFoundError:
                    log(f"Waiting for shared memory to be created: {self.sharedMemoryID}")
                    time.sleep(0.5)
            
        self.pausedManager = PauseManager(paused_shared_memory_id)
        self.isPaused = False
        self.stop = False

    def realTimePrint(self, data):
        data = str(data)
        # Clear the last line
        sys.stdout.write("\r" + " " * self.last_length)
        sys.stdout.flush()

        # Write the new line
        sys.stdout.write("\r" + data)
        sys.stdout.flush()

        # Update the length of the last printed line
        self.last_length = len(data)

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

    def setPreviewFrame(self, frame):
        self.previewFrame = frame if not self.hdr_mode else hdr_to_sdr(frame, self.croppedOutputWidth, self.croppedOututHeight)

    def setFramesRendered(self, framesRendered: int):
        self.framesRendered = framesRendered

    def stopWriting(self):
        self.stop = True

    def writeOutInformation(self):
        """
        fcs = framechunksize
        """
        # Create a shared memory block
        if self.sharedMemoryID is not None:
            log(f"Shared memory name: {self.shm.name}")
        i = 0
        while not self.stop:
            
            if self.previewFrame is not None and self.framesRendered > 0:
                # print out data to stdout
                fps = round(self.framesRendered / (time.time() - self.startTime))
                eta = self.calculateETA(framesRendered=self.framesRendered)
                message = f"FPS: {fps} Current Frame: {self.framesRendered} ETA: {eta}"
                if i == 0:
                    print("\n", file=sys.stderr)
                    i = 1
                self.realTimePrint(message)
                if self.sharedMemoryID is not None and self.previewFrame is not None:
                    # Update the shared array
                    if self.border_detect:
                        padded_frame = padFrame(
                            self.previewFrame,
                            self.width,
                            self.height,
                            self.croppedOutputWidth,
                            self.croppedOututHeight,
                        )
                        try:
                            self.shm.buf[:self.sharedMemoryChunkSize] = bytes(padded_frame)
                        except Exception:
                            pass
                    else:
                        try:
                            self.shm.buf[:self.sharedMemoryChunkSize] = bytes(self.previewFrame)
                        except Exception:
                            pass
                self.isPaused = self.pausedManager.pause_manager()
            time.sleep(0.5) # setting this to a higher value will reduce the cpu usage, and increase fps

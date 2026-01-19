import time
from multiprocessing import shared_memory
from .Util import log


class PauseManager:
    def __init__(self, paused_shared_memory_id):
        self.isPaused = False
        self.prevState = None
        self.paused_shared_memory_id = paused_shared_memory_id
        if self.paused_shared_memory_id is not None:
            while True:
                try:
                    self.pausedSharedMemory = shared_memory.SharedMemory(
                        name=self.paused_shared_memory_id
                    )
                    break
                except FileNotFoundError:
                    log(
                        f"Waiting for shared memory to be created: {self.paused_shared_memory_id}"
                    )
                    time.sleep(0.5)

    def pause_manager(self):
        if self.paused_shared_memory_id is not None:
            return self.pausedSharedMemory.buf[0] == 1
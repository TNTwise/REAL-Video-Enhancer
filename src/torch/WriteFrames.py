import subprocess
import numpy as np
import sys
from queue import Queue
class WriteFrames:
    def __init__(self,
                 width,
                 height,
                 fps
                 ):
        self.width = width
        self.height = height
        self.fps = fps
        self.queueSize = 50
    def buildFFmpeg(self):
        
        self.command = [
            'ffmpeg',
            '-i',
            '-',
            '-r',
            f'{self.fps}',
            '-s',
            f'{self.width}x{self.height}'
            '-f',
            'rawvideo',
            'out.mp4'
        ]
    def start(self,verbose: bool = False, queue: Queue = None):
        self.writeBuffer = queue if queue is not None else Queue(maxsize=self.queueSize)
        self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=sys.stdout,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        writtenFrames = 0
        self.isWritingDone = False
        while True:
                frame = self.writeBuffer.get()
                if frame is None:
                   

                    self.process.stdin.close()
                    self.process.wait()
                    self.isWritingDone = True
                    break

                frame = np.ascontiguousarray(frame)
                self.process.stdin.buffer.write(frame.tobytes())
                writtenFrames += 1

    def write(self, frame: np.ndarray):
        """
        Add a frame to the queue. Must be in RGB format.
        """
        self.writeBuffer.put(frame)
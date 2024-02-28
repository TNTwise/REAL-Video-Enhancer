from rife.rife import *
from WriteFrames import *
from threading import Thread
class Interp:
    def __init__(self):
        self.interpolate_factor = 2
        self.interpolate_process = Rife( interpolation_factor = 2,
        interpolate_method = 'rife4.14',
        width=1280,
        height=720,
        half=True)
        self.interpolate_method = 'rife4.14'

        self.writeBuffer = WriteFrames( fps=60,
       
        width=1280,
        height=720,
        )

        self.writeBuffer.buildFFmpeg()

        writeThread = Thread(target=self.writeBuffer.start)
        writeThread.start()
    def processFrame(self, frame):
        
        try:
            

            
            try:
                if self.prevFrame is not None:
                    self.interpolate_process.run(self.prevFrame, frame)
                    for i in range(2 - 1):
                        result = self.interpolate_process.make_inference(
                            (i + 1) * 1.0 / (self.interpolate_factor + 1)
                        )
            except:
                pass
                    
            self.prevFrame = frame
            self.writeBuffer.write(frame)
        except Exception as e:
            print(f"Something went wrong while processing the frames, {e}")
        finally:
            return 0


    
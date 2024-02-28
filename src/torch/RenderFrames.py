from rife.rife import *

class Interp:
    def __init__(self):
        self.interpolate_factor = 2
        self.interpolate_process = Rife( interpolation_factor = 2,
        interpolate_method = 'rife4.14',
        width=1280,
        height=720,
        half=True)
        self.interpolate_method = 'rife4.14'
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

        except Exception as e:
            print(f"Something went wrong while processing the frames, {e}")
        finally:
            return 0


    
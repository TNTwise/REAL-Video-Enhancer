import numpy as np
class NPMeanSequential:
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect and upscale skip
    """
    def __init__(self):
        self.i0 = None
        self.i1 = None
    def forward(self, img1): #forward method calculates out mean of new img, and saves current img as old
        if self.i0 is None:
            self.i0 = img1
            self.image0mean = np.mean(self.i0)
            return
        self.i1 = img1
        self.img1mean = np.mean(self.i1)

    def isSceneChange(self):
        if self.image0mean > self.img1mean + 20 or self.image0mean < self.img1mean - 20:
            self.image0mean = self.img1mean
            return True
        self.image0mean = self.img1mean
        return False
    
    def isMeanEqual(self):
        return self.image0mean == self.img1mean
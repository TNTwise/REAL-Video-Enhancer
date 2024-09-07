import numpy as np
class NPMeanSequential:
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect and upscale skip
    """
    def __init__(self):
        self.i0 = None
        self.i1 = None
    def sceneDetect(self, img1):
        if self.i0 is None:
            self.i0 = img1
            self.image0mean = np.mean(self.i0)
            return
        self.i1 = img1
        img1mean = np.mean(self.i1)
        if self.image0mean > img1mean + 20 or self.image0mean < img1mean - 20:
            self.image0mean = img1mean
            return True
        self.image0mean = img1mean
        return False
    
    def isEqualImages(self,img1):
        if self.i0 is None:
            self.i0 = img1
            self.image0mean = np.mean(self.i0)
            return
        self.i1 = img1
        img1mean = np.mean(self.i1)
        if (self.i0==self.i1).all():
            self.image0mean = img1mean
            self.i0 = self.i1
            return True
        self.i0 = self.i1
        self.image0mean = img1mean
        return False


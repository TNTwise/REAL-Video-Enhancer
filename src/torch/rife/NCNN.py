from rife_ncnn_vulkan_python import Rife
from src.programData.thisdir import thisdir as ts
import os
import numpy as np
thisdir = ts()
class RifeNCNN:
    def __init__(self,
                 interpolation_factor,
                 interpolate_method,
                 width,
                 height,
                 ensemble,
                 half,
                 threads=2,
                 ncnn_gpu=0,):
        
        self.interpolation_factor = interpolation_factor
        self.interpolation_method = interpolate_method
        self.width = width
        self.height = height
        self.ensemble = ensemble
        self.half = half
        self.handleModel()
        self.createInterpolation(ncnn_gpu=ncnn_gpu,threads=threads)
    def handleModel(self):
        self.modelPath = os.path.join(thisdir,"models","rife",self.interpolation_method)
    def createInterpolation(self,
                            ncnn_gpu=0,
                            threads=2):
        self.render = Rife(gpuid=ncnn_gpu,
                           num_threads=threads,
                           model=self.modelPath,
                           uhd_mode=False)
    def bytesToNpArray(self,bytes):
        return np.ascontiguousarray(
                np.frombuffer(bytes,dtype=np.uint8).reshape(self.height,self.width,3)
            )
    def run1(self,i0,i1):
        self.i0 = self.bytesToNpArray(i0)
        self.i1 = self.bytesToNpArray(i1)
    def make_inference(self,n):
        return np.ascontiguousarray(self.render.process_cv2(self.i0,self.i1,timestep=n))
        
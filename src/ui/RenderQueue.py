from PySide6.QtWidgets import QListWidget
from dataclasses import dataclass


@dataclass
class RenderOptions:
    def __init__(
        self,
        inputFile: str,
        outputPath: str,
        videoWidth: int,
        videoHeight: int,
        videoFps: int,
        tilingEnabled: bool,
        tilesize: str,
        videoFrameCount: int,
        backend: str,
        modelScale: int,
        interpolateTimes: int,
        benchmarkMode: bool,
        sloMoMode: bool,
        dyanmicScaleOpticalFlow: bool,
        ensemble: bool,
        interpolateModelFile: str | None = None,
        upscaleModelFile: str | None = None,
        upscaleModelArch: str | None = None,
        deblurModelFile: str | None = None,
        denoiseModelFile: str | None = None,
        decompressModelFile: str | None = None,
        sceneChangeModelFile: str | None = None,
        hdrMode: bool = False,
        startTime: float | None = None,
        endTime: float | None = None,
        isPreview: bool = False,
        overrideUpscaleScale: int | None = None,
        encoderCommand: str | None = None,
    ):
        self._inputFile = inputFile
        self._outputPath = outputPath
        self._videoWidth = videoWidth
        self._videoHeight = videoHeight
        self._videoFps = videoFps
        self._tilingEnabled = tilingEnabled
        self._tilesize = tilesize
        self._videoFrameCount = videoFrameCount
        self._backend = backend
        self._interpolateModelFile = interpolateModelFile
        self._deblurModelFile = deblurModelFile
        self._upscaleModelFile = upscaleModelFile
        self._upscaleModelArch = upscaleModelArch
        self._denoiseModelFile = denoiseModelFile
        self._decompressModelFile = decompressModelFile
        self._sceneChangeModelFile = sceneChangeModelFile
        self._modelScale = modelScale
        self._interpolateTimes = interpolateTimes
        self._benchmarkMode = benchmarkMode
        self._sloMoMode = sloMoMode
        self._dyanmicScaleOpticalFlow = dyanmicScaleOpticalFlow
        self._ensemble = ensemble
        self._hdrMode = hdrMode
        self._startTime = startTime
        self._endTime = endTime
        self._isPreview = isPreview
        self._overrideUpscaleScale = overrideUpscaleScale
        self._encoderCommand = encoderCommand

    @property
    def inputFile(self):
        return self._inputFile

    @inputFile.setter
    def inputFile(self, value: str):
        self._inputFile = value

    @property
    def outputPath(self):
        return self._outputPath

    @outputPath.setter
    def outputPath(self, value: str):
        self._outputPath = value

    @property
    def videoWidth(self):
        return self._videoWidth

    @videoWidth.setter
    def videoWidth(self, value: int):
        self._videoWidth = value

    @property
    def videoHeight(self):
        return self._videoHeight

    @videoHeight.setter
    def videoHeight(self, value: int):
        self._videoHeight = value

    @property
    def videoFps(self):
        return self._videoFps

    @videoFps.setter
    def videoFps(self, value: int):
        self._videoFps = value

    @property
    def tilingEnabled(self):
        return self._tilingEnabled

    @tilingEnabled.setter
    def tilingEnabled(self, value: bool):
        self._tilingEnabled = value

    @property
    def tilesize(self):
        return self._tilesize

    @tilesize.setter
    def tilesize(self, value: str):
        self._tilesize = value

    @property
    def videoFrameCount(self):
        return self._videoFrameCount

    @videoFrameCount.setter
    def videoFrameCount(self, value: int):
        self._videoFrameCount = value

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        self._backend = value
    
    @property
    def modelScale(self):
        return self._modelScale
    @modelScale.setter
    def modelScale(self, value: int):
        self._modelScale = value

    @property
    def denoiseModelFile(self):
        return self._denoiseModelFile
    @denoiseModelFile.setter
    def denoiseModelFile(self, value: str):
        self._denoiseModelFile = value
    
    @property
    def decompressModelFile(self):
        return self._decompressModelFile
    @decompressModelFile.setter
    def decompressModelFile(self, value: str):
        self._decompressModelFile = value
    
    @property
    def sceneChangeModelFile(self):
        return self._sceneChangeModelFile
    @sceneChangeModelFile.setter
    def sceneChangeModelFile(self, value: str):
        self._sceneChangeModelFile = value

    @property
    def upscaleModelArch(self):
        return self._upscaleModelArch
    @upscaleModelArch.setter
    def upscaleModelArch(self, value: str):
        self._upscaleModelArch = value

    @property
    def interpolateModelFile(self):
        return self._interpolateModelFile

    @interpolateModelFile.setter
    def interpolateModelFile(self, value: str):
        self._interpolateModelFile = value
    
    @property
    def deblurModelFile(self):
        return self._deblurModelFile
    
    @deblurModelFile.setter
    def deblurModelFile(self, value: str):
        self._deblurModelFile = value

    @property
    def upscaleModel(self):
        return self._upscaleModel

    @upscaleModel.setter
    def upscaleModel(self, value: str):
        self._upscaleModel = value

   
   
    @property
    def interpolateTimes(self):
        return self._interpolateTimes

    @interpolateTimes.setter
    def interpolateTimes(self, value: int):
        self._interpolateTimes = value

    @property
    def benchmarkMode(self):
        return self._benchmarkMode

    @benchmarkMode.setter
    def benchmarkMode(self, value: bool):
        self._benchmarkMode = value

    @property
    def sloMoMode(self):
        return self._sloMoMode

    @sloMoMode.setter
    def sloMoMode(self, value: bool):
        self._sloMoMode = value

    @property
    def dyanmicScaleOpticalFlow(self):
        return self._dyanmicScaleOpticalFlow

    @dyanmicScaleOpticalFlow.setter
    def dyanmicScaleOpticalFlow(self, value: bool):
        self._dyanmicScaleOpticalFlow = value

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, value: bool):
        self._ensemble = value

    @property
    def upscaleModelFile(self):
        return self._upscaleModelFile

    @upscaleModelFile.setter
    def upscaleModelFile(self, value: str):
        self._upscaleModelFile = value
    
    @property
    def hdrMode(self):
        return self._hdrMode
    @hdrMode.setter
    def hdrMode(self, value: bool):
        self._hdrMode = value
    
    @property
    def startTime(self):
        return self._startTime
    @startTime.setter
    def startTime(self, value: float):
        self._startTime = value

    @property
    def endTime(self):
        return self._endTime
    @endTime.setter
    def endTime(self, value: float):
        self._endTime = value

    @property
    def isPreview(self):
        return self._isPreview
    @isPreview.setter
    def isPreview(self, value: bool):
        self._isPreview = value
    
    @property
    def overrideUpscaleScale(self):
        return self._overrideUpscaleScale
    @overrideUpscaleScale.setter
    def overrideUpscaleScale(self, value: int):
        self._overrideUpscaleScale = value
    
    @property
    def encoderCommand(self):
        return self._encoderCommand
    @encoderCommand.setter
    def encoderCommand(self, value: str):
        self._encoderCommand = value
        
        


class RenderQueue:
    def __init__(self, qlistwidget: QListWidget):
        self.queue = []
        self.inputNameList = []  # this list links up 1:1 with the queue, storing input names allow for index searching
        self.qlistwidget = qlistwidget

    def add(self, renderable: RenderOptions):
        self.queue.append(renderable)
        self.qlistwidget.addItem(renderable.inputFile)
        self.inputNameList.append(renderable.inputFile)
    
    def addToStart(self, renderable: RenderOptions):
        self.queue.insert(0, renderable)
        self.qlistwidget.insertItem(0, renderable.inputFile)
        self.inputNameList.insert(0, renderable.inputFile)

    def clear(self):
        self.queue.clear()
        self.inputNameList.clear()
        self.qlistwidget.clear()

    def getQueue(self) -> list[RenderOptions]:
        return self.queue

    def _swapListPositions(self, list1, index1, index2):
        list1[index1], list1[index2] = (
            list1[index2],
            list1[index1],
        )  # flip

    def remove(self):
        try:
            index = self.qlistwidget.currentRow()
            del self.queue[index]
            del self.inputNameList[index]
            self.qlistwidget.takeItem(index)  # remove the item from the list widget
        except IndexError:
            pass

    def moveitem(self, direction="up"):
        try:
            index = self.qlistwidget.currentRow()
            if direction == "down":
                new_index = index + 1
            else:
                new_index = index - 1
            self._swapListPositions(self.queue, index, new_index)
            self._swapListPositions(
                self.inputNameList, index, new_index
            )  # swap in index

            currentItem = self.qlistwidget.takeItem(index)
            self.qlistwidget.insertItem(new_index, currentItem)  # swap in gui
            self.qlistwidget.setCurrentRow(new_index)
        except Exception:  # catches out of index errors, not that much of an issue
            pass

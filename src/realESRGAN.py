def start(self):
    self.ui.ETAPreview.setText('ETA:')
    self.ui.processedPreview.setText('Files Processed:')
    self.runPB(self.videoName,1)
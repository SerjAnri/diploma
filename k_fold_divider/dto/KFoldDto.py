class KFoldDto:
    def __init__(self, trainFiles, kFoldNumber):
        self.trainFiles = trainFiles
        # self.valFiles = valFiles
        self.kFoldNumber = kFoldNumber
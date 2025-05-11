from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pandas as pd

from dto.KFoldDto import KFoldDto


class KFoldDividerService:
    def __init__(self, labelsPath: str, k: int):
        self.labelsPath = labelsPath
        self.k = k

    def generateKFolds(self, labelId: str = 'id', labelName: str = 'label') -> [KFoldDto]:
        skf = StratifiedShuffleSplit(n_splits=self.k, train_size=0.5, test_size=0.5)
        labelsDf = pd.read_csv(self.labelsPath)
        X = labelsDf[labelId]
        y = labelsDf[labelName]
        resultList = []
        for fold, (trainIndex, valIndex) in enumerate(skf.split(X, y)):
            trainFiles = X.iloc[trainIndex].to_list()
            # valFiles = X.iloc[valIndex].to_list()
            resultList.append(
                KFoldDto(trainFiles=trainFiles, kFoldNumber=fold)
            )
            # resultList.append(
            #     KFoldDto(trainFiles=valFiles, kFoldNumber=fold + 1)
        #     # )
        return resultList
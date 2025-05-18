from sklearn.model_selection import KFold
import torchvision as tv
from dto.KFoldDto import KFoldDto


class KFoldDividerService:
    def __init__(self, k: int):
        self.k = k

    def generateKFolds(self) -> [KFoldDto]:
        transform = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
        ])
        train_ds = tv.datasets.CIFAR10(root='C:\\Users\\Asus\\Desktop\\pythonProject\\data', train=True, download=False, transform=transform)
        # train_ds = tv.datasets.CIFAR100(root='C:\\Users\\Asus\\Desktop\\pythonProject\\data', train=True, download=False, transform=transform)
        kf = KFold(n_splits=self.k, shuffle=False, random_state=None)
        resultList = []
        for fold, (trainIndex, valIndex) in enumerate(kf.split(list(range(len(train_ds))))):
            resultList.append(
                KFoldDto(trainFiles=trainIndex.tolist(), kFoldNumber=fold)
            )
        return resultList
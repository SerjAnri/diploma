from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose
from JSONTransformer import KFoldDto


class DataLoaderService:
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path
    def prepareTrainDataLoaders(self, train_tfm: Compose, dto: KFoldDto) -> DataLoader:
        train_ds = datasets.CIFAR100(root=self.train_path, train=True, download=False, transform=train_tfm)
        train_ds.data = train_ds.data[dto.trainFiles]
        print(len(train_ds.data))
        batch_size = 400
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
        return train_dl

    def prepareValDataLoaders(self, val_tfm: Compose) -> DataLoader:
        val_ds = datasets.CIFAR100(root=self.val_path, train=False, download=False, transform=val_tfm)
        batch_size = 400
        val_dl = DataLoader(val_ds, batch_size, num_workers=3, pin_memory=True)
        return val_dl

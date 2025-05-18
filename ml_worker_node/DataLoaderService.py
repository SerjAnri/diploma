from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import Compose


class DataLoaderService:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def prepareTrainDataLoaders(self, train_tfm: Compose, dto) -> DataLoader:
        # train_ds = datasets.CIFAR100(root=self.dataset_path, train=True, download=False, transform=train_tfm)
        train_ds = datasets.CIFAR10(root=self.dataset_path, train=True, download=False, transform=train_tfm)
        sampler = SubsetRandomSampler(dto["trainFiles"])
        batch_size = 256
        train_dl = DataLoader(train_ds, batch_size, sampler=sampler)
        return train_dl

    def prepareValDataLoaders(self, val_tfm: Compose) -> DataLoader:
        # val_ds = datasets.CIFAR100(root=self.val_path, train=False, download=False, transform=val_tfm)
        val_ds = datasets.CIFAR10(root=self.dataset_path, train=False, download=False, transform=val_tfm)
        batch_size = 256
        val_dl = DataLoader(val_ds, batch_size)
        return val_dl

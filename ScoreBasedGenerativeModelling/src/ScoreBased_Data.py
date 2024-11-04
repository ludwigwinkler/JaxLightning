import os, sys
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torchvision
import torchvision.datasets
from torch.utils.data import DataLoader

filepath = Path().resolve()
phd_path = Path(str(filepath).split("PhD")[0] + "PhD")
jax_path = Path(str(filepath).split("Jax")[0] + "Jax")
score_path = Path(str(filepath).split("Jax")[0] + "Jax/ScoreBasedGenerativeModelling")
sys.path.append(str(phd_path))


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class MNIST(torchvision.datasets.MNIST):
    """
    Numpy version of MNIST
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int):
        """
        Args:
                index (int): Index

        Returns:
                tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], int(self.targets[index])

        img = img.numpy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: x / 255),
                torchvision.transforms.Lambda(lambda x: x.reshape(1, 28, 28)),
            ]
        )

    def prepare_data(self) -> None:
        if not os.path.isdir(path := score_path / "data"):
            torchvision.datasets.MNIST(root=path, download=True)

    def setup(self, stage=None):
        data_path = score_path / "data"

        self.train_data = MNIST(root=data_path, train=True, transform=self.transform)
        self.val_data = MNIST(root=data_path, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=64, shuffle=True, collate_fn=numpy_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=64, shuffle=True, collate_fn=numpy_collate
        )

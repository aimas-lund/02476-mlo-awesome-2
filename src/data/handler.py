import pickle
from pathlib import Path

import numpy as np
import torch
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from src.data import _PATH_DATA


class CIFAR10Dataset(Dataset):
    def __init__(
        self, train: bool, x_dim: int = 32, y_dim: int = 32, col_dim: int = 3
    ) -> None:
        content = []

        if train:
            data_files = [
                Path(_PATH_DATA) / "cifar-10" / f"data_batch_{i}" for i in range(1, 6)
            ]
        else:
            data_files = [Path(_PATH_DATA) / "cifar-10" / "test_batch"]

        for file in data_files:
            with open(file, "rb") as fo:
                data_dict = pickle.load(fo, encoding="bytes")

            # clean-up labels of loaded pickle data
            clean_dict = {}
            for key in data_dict:
                clean_dict[key.decode()] = data_dict[key]
            content.append(clean_dict)

        self.data = torch.tensor(np.concatenate([c["data"] for c in content])).reshape(
            -1, col_dim, x_dim, y_dim
        )
        self.targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    cifar10_normalization(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    cifar10_normalization(),
                ]
            )

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int):
        data, target = self.data[idx].float(), self.targets[idx]
        if self.transform:
            data = self.transform(data)
        return data, target

import pickle
from pathlib import Path

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from typing import Tuple, Any

from src.data import _PATH_DATA


class CIFAR10Dataset(Dataset):
    """
    A class inheriting from the torch Dataset class. It is designed to fetch and 
    format the raw CIFAR10 dataset in the /data/cifar10 directory to a 4D torch Tensor.
    In addition, it also specifies transformations that is performed on the training data after formatting.
    """
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

        self.data = (
            torch.tensor(np.concatenate([c["data"] for c in content]))
            .reshape(-1, col_dim, x_dim, y_dim)
            .float()
        )
        self.targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
        self.N = self.data.size()[0]

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.data.mean(dim=(0, 2, 3)), self.data.std(dim=(0, 2, 3))
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.data.mean(dim=(0, 2, 3)), self.data.std(dim=(0, 2, 3))
                    ),
                ]
            )

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        data, target = self.data[idx].float(), self.targets[idx]
        if self.transform:
            data = self.transform(data)
        return data, target

    def generate_sample(self, idx: int) -> None:
        image = self.data[idx,:,:,:]
        file_name = os.path.join(_PATH_DATA, "sample/sample.png")
        save_image(image, file_name)


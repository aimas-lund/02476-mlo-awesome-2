import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from src.data import _PATH_DATA


class CIFAR10Dataset(Dataset):

    def __init__(self, train: bool, x_dim: int=32, y_dim: int=32, col_dim: int=3) -> None:
        content = []

        if train:
            data_files = [Path(_PATH_DATA) / "cifar-10" / f"data_batch_{i}" for i in range(1, 6)]
        else:
            data_files = [Path(_PATH_DATA) / "cifar-10" / "test_batch"]

        for file in data_files:
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            
            # clean-up labels of loaded pickle data
            clean_dict = {}
            for key in data_dict:
                clean_dict[key.decode()] = data_dict[key]
            content.append(clean_dict)

        self.data = torch.tensor(np.concatenate([c['data'] for c in content])).reshape(-1, col_dim, x_dim, y_dim)
        self.targets = torch.tensor(np.concatenate([c['labels'] for c in content]))

    
    def __len__(self) -> int:
        return self.targets.numel()
    
    def __getitem__(self, idx: int):
        return self.data[idx].float(), self.targets[idx]

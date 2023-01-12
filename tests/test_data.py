import os

import pytest
import torch

from src.data.handler import CIFAR10Dataset
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data folder does not exist")
def test_cifar10_data_dimensions():
    train_set = CIFAR10Dataset(train=True)
    test_set = CIFAR10Dataset(train=False)

    assert train_set.data.size() == torch.empty(50000, 3, 32, 32).size(), "Expected a training set of 50000 instances with 32x32x3 dimensions."
    assert test_set.data.size() == torch.empty(10000, 3, 32, 32).size(), "Expected a training set of 10000 instances with 32x32x3 dimensions."


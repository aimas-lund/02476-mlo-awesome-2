import pytest

from src.models import train_model
from tests import _PATH_MODELS
from tests.utils import ModelTestContext
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra


def test_resnet_training_without_error():
    context = ModelTestContext('resnet10t')

    context.start_test()
    # GlobalHydra.instance().clear()
    with initialize(config_path="../src/models"):
        try:
            train_model.train_model()
            assert True
        except Exception as e:
            assert False, f"Exception raised during training: {e}"
    context.stop_test()

import pytest

from src.models import train_model
from tests import _PATH_MODELS
from tests.utils import ModelTestContext
from hydra import initialize, compose


def test_resnet10_training():
    context = ModelTestContext('resnet10t')

    context.start_test()
    with initialize(version_base=None, config_path="../src/models"):
        try:
            cfg = compose(config_name="config.yaml")
            train_model.train_model(cfg)
            assert True
        except Exception as e:
            assert False, f"Exception raised during training: {e}"
    context.stop_test()

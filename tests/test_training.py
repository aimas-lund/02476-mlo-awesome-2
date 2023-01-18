import pytest
from hydra import compose, initialize
from src.models import train_model

from tests import _PATH_MODELS
from tests.utils import ModelTestContext


def test_resnet10_training():
    ## A end-to-end model training test of the resnet10 model, using hydra.
    context = ModelTestContext("resnet10t")

    context.start_test()
    with initialize(version_base=None, config_path="./"):
        try:
            cfg = compose(config_name="config.yaml")
            train_model.train_model(cfg)
            assert True
        except Exception as e:
            assert False, f"Exception raised during training: {e}"
    context.stop_test()


@pytest.mark.xfail  # wandb has some unexpected behaviour which is not resolved easily
def test_unsupported_model_training():
    ## An edge-case test where an unsupported model is specified to be trained.
    context = ModelTestContext("Ub3Rl33TM0d31")

    context.start_test()
    with pytest.raises(RuntimeError):
        with initialize(version_base=None, config_path="./"):
            cfg = compose(config_name="config.yaml")
            train_model.train_model(cfg)
    context.stop_test()

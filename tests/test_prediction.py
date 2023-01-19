import pytest

from cloud.functions.predict.main import (
    _get_newest_checkpoint_path,
    generate_model,
    mlops_predict,
)


def test_model():
    model = generate_model("resnet10t", test=True)

    assert model != None


def test_get_newest_checkpoint_path():
    checkpoints = [
        "resnet10t-2023-01-18-22-18-01.pth",
        "resnet10t-2023-01-19-22-18-01.pth",
        "resnet10t-2023-02-18-22-18-01.pth",
    ]

    newest_checkpoint = _get_newest_checkpoint_path(checkpoints)

    assert newest_checkpoint == checkpoints[2], "Newest checkpoint was not found."

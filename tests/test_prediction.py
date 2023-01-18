import pytest
import torch

from tests import _PATH_MODELS


@pytest.mark.xfail
def test_input_output_dims():
    ## Test the expected in- and output dimensions of the model
    model = None  # TODO: replace with model implementation
    expected_output_dim = 10
    expected_input_dim = torch.empty(1, 32, 32, 3).size()


    assert False


@pytest.mark.xfail
def test_model_input_error():
    ## Test a an input with an invalid dimension
    model = None  # TODO: replace with model implementation
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model(torch.randn(1, 2, 3))

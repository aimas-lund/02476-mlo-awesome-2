import pytest
import torch


@pytest.mark.xfail
def test_input_output_dims():
    model = None    # TODO: replace with model implementation
    expected_output_dim = 10
    expected_input_dim = torch.empty(1, 32, 32, 3).size()

    ## Perform model prediction
    # model.prediction = something

    assert False


@pytest.mark.xfail
def test_model_input_error():
    model = None    # TODO: replace with model implementation
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model(torch.randn(1, 2, 3))
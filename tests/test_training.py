from src.models import train_model

@pytest.mark.xfail
def test_training():
    train_model.run()

    assert False
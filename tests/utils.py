import yaml
import os
from typing import List, Any
import logging

from tests import _PATH_MODELS, _DIR_ROOT, _PATH_TRAINED_MODELS


class ModelTestContext():

    log = logging.getLogger(__name__)

    def __init__(self, model_type: str, config_filename: str="config.yaml") -> None:
        self.model_type = model_type
        self.original_config = None
        self.config_file = _PATH_MODELS / config_filename
        self.test_config_file = _DIR_ROOT / config_filename

        # create a test config file in the tests directory, if it doesn't exist
        if not os.path.exists(self.test_config_file.resolve()):
            open(self.test_config_file.resolve(), "w").close()

        # read the existing config file
        try:
            with open(self.config_file.resolve(), "r") as file:
                self.original_config = yaml.safe_load(file)
        except FileNotFoundError as e:
            self.log.warn("Config file for model training is not found.")

        self.test_config = {
            "params": {
                "model": f"{self.model_type}",
                "epochs": 1,
                "in_chans": 3,
                "num_classes": 10,
                "learning_rate": 1e-3,
                "batch_size": 32,
            }
        }

    def _write_config_file(self, restore_config: bool=False) -> None:

        content = self.original_config if restore_config else self.test_config

        with open(self.test_config_file, "w") as f:
            yaml.dump(content, f, default_flow_style=False)

    def _file_cleanup(self) -> None:
        # delete test config file
        os.remove(self.test_config_file)

        # delete saves weights and biases from training
        files = [file for file in os.listdir(_PATH_TRAINED_MODELS.resolve()) if file.endswith(".pth")]

        for f in files:
            os.remove(os.path.join(_PATH_TRAINED_MODELS.resolve(), f))
        

    def start_test(self) -> None:
        self._write_config_file(restore_config=False)


    def stop_test(self) -> None:
        self._file_cleanup()




if __name__ == "__main__":
    context = ModelTestContext("resnet10t")
    context.start_test()
    context.stop_test()
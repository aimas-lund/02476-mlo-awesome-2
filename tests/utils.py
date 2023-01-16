import yaml
from typing import List, Any

from tests import _PATH_MODELS


class ModelTestContext():
    def __init__(self, model_type: str, config_filename: str="config.yaml") -> None:
        self.model_type = model_type
        self.original_config = None
        self.config_file = _PATH_MODELS / config_filename

        with open(self.config_file.resolve(), "r") as file:
            self.original_config = yaml.safe_load(file)

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

    def _change_config(self, restore_config: bool=False) -> None:

        content = self.original_config if restore_config else self.test_config

        with open(self.config_file.resolve(), "w") as file:
            yaml.dump(content, file, default_flow_style=False)
        

    def start_test(self) -> None:
        self._change_config(restore_config=False)


    def stop_test(self) -> None:
        self._change_config(restore_config=True)


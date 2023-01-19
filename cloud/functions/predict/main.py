import logging
import os
from datetime import datetime
from io import BytesIO
from typing import List

import torch
from google.cloud import storage
from PIL import Image
from timm import create_model
from torchvision import transforms

log = logging.getLogger(__name__)


class PredictModel:
    def __init__(self, model_name: str, num_classes: int, checkpoint_path: str) -> None:

        d = torch.device("cpu")
        self.model_name = model_name
        self.model = (
            create_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=False,
                checkpoint_path=checkpoint_path,
            )
            .to(d)
            .eval()
        )

        self.class_mapping = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }

    def predict(self, img: torch.Tensor) -> int:
        predictions = self.model(img)
        return predictions


def mlops_predict(request):
    """
    Google Cloud Function exposed to the internet. The function takes an HTTP POST request with an attached image
    and returns a string indicating the classification of the image.

    Args:
        request: An HTTP request containing an 'image' file.

    Returns:
        A string indicating the classification of the image.
    """
    model_name = "resnet10t"

    # read and format image file to a torch Tensor
    image_content = request.files.get("image")
    transform = transforms.Compose([transforms.ToTensor()])

    with Image.open(BytesIO(image_content.read())) as img:
        input_tensor = transform(img)

    log.info("Found checkpoints for the specified model.")
    log.info("Initializing model...")

    # get newest model checkpoint from Google Cloud Bucket
    client = storage.Client()
    bucket_name = "mlops-checkpoints"
    bucket = client.get_bucket(bucket_name)
    files = bucket.list_blobs()

    checkpoints = [file.name for file in files]

    file_name = _get_newest_checkpoint_path(checkpoints, model_name=model_name)

    # create a tmp folder, which is writable in Google Cloud Function env.
    os.makedirs("/tmp", exist_ok=True)

    checkpoint_path = f"/tmp/{file_name}"

    blob = bucket.blob(file_name)
    blob.download_to_filename(checkpoint_path)

    # classify the provided image
    model = PredictModel(
        model_name=model_name, num_classes=10, checkpoint_path=checkpoint_path
    )

    log.info(f"Model created with parameters: {model.model.parameters()}")
    input_tensor = input_tensor[None, :, :, :]
    prediction = model.predict(input_tensor)
    pred_idx = torch.argmax(prediction).item()
    predicted_class = model.class_mapping[pred_idx]

    log.info(f"Predicted the class: '{predicted_class}'")

    return f"Predicted class: {predicted_class}"


def _get_newest_checkpoint_path(
    checkpoints: List[str], model_name: str = "resnet10t"
) -> str:
    """
    Fetches the newest checkpoint path for a given model. The function assumes that model checkpoints
    are stored in the format "{model name}-{year}-{month}-{day}-{hour}-{minute}-{second}".

    Args:
         checkpoints: A list of strings containing the checkpoint file names.
         model_name: Name of the model whose checkpoint is searched for, for example "resnet10t".

    Returns:
         Absolute path to the newest checkpoint for the specified model name in the /models directory.
    """

    def filter(checkpoint: str, model_name: str) -> bool:
        return (checkpoint.endswith(".pth")) and (checkpoint.startswith(model_name))

    checkpoints = [
        checkpoint for checkpoint in checkpoints if filter(checkpoint, model_name)
    ]

    if len(checkpoints) < 1:
        return None

    dates = []
    for ch in checkpoints:
        # remove model name and file extension from filename
        datetime_split = (
            ch.replace(f"{checkpoints}-", "").replace(".pth", "").split("-")
        )
        dates.append(
            datetime(
                year=int(datetime_split[0]),
                month=int(datetime_split[1]),
                day=int(datetime_split[2]),
                hour=int(datetime_split[3]),
                minute=int(datetime_split[4]),
                second=int(datetime_split[5]),
            )
        )

    # find the newest checkpoint
    newest_date = max(dates)
    index = dates.index(newest_date)
    newest_checkpoint = checkpoints[index]

    return newest_checkpoint

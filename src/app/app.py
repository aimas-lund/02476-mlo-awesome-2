import logging
import os
from datetime import datetime
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from src.app import _PATH_MODELS
from src.models.predict_model import PredictModel

app = FastAPI()
log = logging.getLogger(__name__)


@app.get("/")
def root():
    """
    Root path of the FastAPI app. Used for health checks
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/prediction")
async def model_prediction(
    data: UploadFile = File(...),
    x_dim: int = 32,
    y_dim: int = 32,
    z_dim: int = 3,
    num_classes: int = 10,
    model_name: str = "resnet10t",
) -> None:
    log.info(
        f'"/prediction"-endpoint was called with the parameters: \
      x_dim: {x_dim}, y_dim: {y_dim}, z_dim: {z_dim}, model_name: {model_name}'
    )

    transform = transforms.Compose([transforms.ToTensor()])
    image_content = await data.read()
    with Image.open(BytesIO(image_content)) as img:
        input_tensor = transform(img)

    checkpoint_path = _get_newest_checkpoint_path(model_name=model_name)
    log.info("Found checkpoints for the specified model.")
    log.info("Initializing model...")

    model = PredictModel(
        model_name=model_name,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )

    log.info(f"Model created with parameters: {model.model.parameters()}")
    input_tensor = input_tensor[None, :, :, :]
    prediction = model.predict(input_tensor)
    pred_idx = torch.argmax(prediction).item()
    predicted_class = model.class_mapping[pred_idx]

    log.info(f"Predicted the class: '{predicted_class}'")
    response = {
        "prediction": predicted_class,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def _get_newest_checkpoint_path(model_name: str) -> Path:
    """
    Fetches the newest checkpoint path for a given model. The function assumes that model checkpoints
    are stored in the format "{model name}-{year}-{month}-{day}-{hour}-{minute}-{second}".

    Args:
       model_name: Name of the model whose checkpoint is searched for, for example "resnet10".

    Returns:
       Absolute path to the newest checkpoint for the specified model name in the /models directory.
    """

    def filter(checkpoint: str, model_name: str) -> bool:
        return (checkpoint.endswith(".pth")) and (checkpoint.startswith(model_name))

    checkpoints = [
        checkpoint
        for checkpoint in os.listdir(_PATH_MODELS.resolve())
        if filter(checkpoint, model_name)
    ]

    if len(checkpoints) < 1:
        return None

    dates = []
    for ch in checkpoints:
        # remove model name and file extension from filename
        datetime_split = ch.replace(f"{model_name}-", "").replace(".pth", "").split("-")
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
    abs_path = _PATH_MODELS / checkpoints[index]

    return abs_path.resolve()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

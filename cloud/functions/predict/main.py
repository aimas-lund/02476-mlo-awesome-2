import logging
import os
from http import HTTPStatus
from io import BytesIO

from google.cloud import storage
from PIL import Image
from timm import create_model
from torch import Tensor, argmax, device
from torchvision import transforms

log = logging.getLogger(__name__)
bucket_name = "mlops-checkpoints"


class PredictModel:
    def __init__(self, model_name: str, num_classes: int, checkpoint_path: str) -> None:

        d = device("cpu")
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

    def predict(self, img: Tensor) -> int:
        predictions = self.model(img)
        return predictions


def mlops_predict_trained(request):
    image_content = request.files.get("image")

    transform = transforms.Compose([transforms.ToTensor()])

    with Image.open(BytesIO(image_content.read())) as img:
        input_tensor = transform(img)

    log.info("Found checkpoints for the specified model.")
    log.info("Initializing model...")

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    file_name = "resnet10t-2023-01-18-22-18-01.pth"

    os.makedirs("/tmp", exist_ok=True)

    checkpoint_path = "/tmp/resnet10t-2023-01-18-22-18-01.pth"

    blob = bucket.blob(file_name)
    blob.download_to_filename(checkpoint_path)

    model = PredictModel(
        model_name="resnet10t", num_classes=10, checkpoint_path=checkpoint_path
    )

    log.info(f"Model created with parameters: {model.model.parameters()}")
    input_tensor = input_tensor[None, :, :, :]
    prediction = model.predict(input_tensor)
    pred_idx = argmax(prediction).item()
    predicted_class = model.class_mapping[pred_idx]

    log.info(f"Predicted the class: '{predicted_class}'")

    return f"Predicted class: {predicted_class}"

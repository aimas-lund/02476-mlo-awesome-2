from torch import device, argmax, Tensor
from io import BytesIO
from timm import create_model
from http import HTTPStatus
from PIL import Image
from torchvision import transforms
import logging

log = logging.getLogger(__name__)

class PredictModel():
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
    ) -> None:
        
        d = device("cpu")
        self.model_name = model_name
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
        ).to(d).eval()

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
            9: "truck"
        }


    def predict(self, img: Tensor) -> int:
        predictions = self.model(img)
        return predictions


def mlops_predict(request):
    image_content = request.files.get('image')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    with Image.open(BytesIO(image_content.read())) as img:
        input_tensor = transform(img)

    log.info("Found checkpoints for the specified model.")
    log.info("Initializing model...")

    model = PredictModel(
        model_name='resnet10t',
        num_classes=10,
    )

    log.info(f"Model created with parameters: {model.model.parameters()}")
    input_tensor = input_tensor[None, :, :, :]
    prediction = model.predict(input_tensor)
    pred_idx = argmax(prediction).item()
    predicted_class = model.class_mapping[pred_idx]

    log.info(f"Predicted the class: \'{predicted_class}\'")
    response = {
        "prediction": predicted_class,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return f"Predicted class: {predicted_class}"

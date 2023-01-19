# from make_dataset3 import cifar10
import numpy as np
import timm
import torch

from src.models import _PATH_MODELS
import wandb
from PIL import Image as im

class_mapping = {
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

def validation(
        model, 
        loss_func, 
        dataloader, 
        device
    ):
    val_loss = 0.0
    val_correct = 0
    size_sampler = len(dataloader.sampler)
    with torch.no_grad():
        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)
            y_hat = model(images)
            loss = loss_func(y_hat, labels.long().squeeze())

            val_loss += loss.item() * images.size(0)
            _, pred = torch.max(y_hat.data, 1)
            val_correct += (pred == labels.long().squeeze()).sum().item()

            # _wandb_log_table(images, labels, pred)

    return np.round(val_loss / size_sampler, 4), np.round(
        val_correct * 100.0 / size_sampler, 3
    )

def _wandb_log_table(images, labels, prediction) -> None:
    table_rows = []

    for idx in range(len(images)):
        img2 = images[idx].reshape(32,32,3).cpu().numpy().astype(np.uint8)
        img2 = im.fromarray(img2, mode="RGB")

        table_row = [wandb.Image(img2),class_mapping[prediction.cpu().numpy()[idx]],class_mapping[labels.long().squeeze().cpu().numpy()[idx]]]
        table_rows.append(table_row)

    pred_table = wandb.Table(data=table_rows, columns=["images","pred","actual"])
    wandb.log({"table": pred_table})


class PredictModel:
    def __init__(self, model_name: str, num_classes: int, checkpoint_path: str) -> None:

        device = torch.device("cpu")
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model = (
            timm.create_model(
                model_name=model_name,
                pretrained=False,
                checkpoint_path=checkpoint_path,
                num_classes=num_classes,
            )
            .to(device)
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

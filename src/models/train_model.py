import logging
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import timm.optim
import torch
from src.data.handler import CIFAR10Dataset
from src.models import _PATH_MODELS, _PATH_VISUALIZATION
from src.models.predict_model import validation
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path="./")
def train(cfg: DictConfig) -> None:
    train_model(cfg)


def train_model(cfg: DictConfig) -> None:
    log.info(f"Running with config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Running on: {device}")

    log.info("Initializing datasets and data loaders")
    training_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    train_split, validation_split = random_split(
        training_dataset, [int(0.9 * training_dataset.N), int(0.1 * training_dataset.N)]
    )

    training_dataloader = DataLoader(
        train_split, batch_size=cfg.params.batch_size, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_split, batch_size=cfg.params.batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.params.batch_size, shuffle=True
    )

    log.info(f"Initializing model: {cfg.params.model}")
    model = timm.create_model(
        cfg.params.model,
        pretrained=True,
        in_chans=cfg.params.in_chans,
        num_classes=cfg.params.num_classes,
    )

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.params.learning_rate)
    loss_func = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history, best_model = _initiate_training(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        scheduler=scheduler,
        train_loader=training_dataloader,
        val_loader=validation_dataloader,
        epochs=cfg.params.epochs,
        device=device,
    )

    # Saving model state dict
    datetime_now = datetime.now()
    datetime_str = datetime_now.strftime("%m-%d-%Y-%H-%M-%S")
    state_dict_filename = cfg.params.model + "-" + datetime_str + ".pth"
    state_dict_path = _PATH_MODELS / state_dict_filename
    log.info(f'Saving state dict as "{state_dict_filename}" ')
    torch.save(best_model.state_dict(), state_dict_path.resolve())

    # ploting results
    plot_history(history)
    torch.load(state_dict_path)
    test_loss, test_accuracy = validation(
        best_model, loss_func, test_dataloader, device
    )
    log.info(f"test loss: {test_loss}, test accuracy: {test_accuracy}")


# Training Function
def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Any, Any]:
    train_loss = 0.0
    train_correct = 0
    size_sampler = len(train_loader.sampler)

    for images, labels in train_loader:

        # Pushing to device (cuda or CPU)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        y_hat = model(images)

        # Compute loss
        loss = loss_func(y_hat, labels.long().squeeze())

        loss.backward()
        optimizer.step()

        # loss and correct values compute
        train_loss += loss.item() * images.size(0)
        _, pred = torch.max(y_hat.data, 1)
        train_correct += sum(pred == labels.long().squeeze()).sum().item()

    return np.round(train_loss / size_sampler, 4), np.round(
        train_correct * 100.0 / size_sampler, 3
    )


# running the model
def _initiate_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    scheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> Tuple[Dict[str, List[Any]], torch.nn.Module]:

    best_acc = 0
    log.info("Initializing training...")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for e in range(epochs):

        train_loss, train_acc = _training_step(model, optimizer, loss_func, train_loader, device)
        val_loss, val_acc = validation(model, loss_func, val_loader, device)

        scheduler.step()

        if val_acc > best_acc:
            log.info(
                f">> Saving Best Model with Val Acc: Old: {best_acc} | New: {val_acc}"
            )
            best_model = deepcopy(model)
            best_acc = val_acc

        if (e + 1) % 2 == 0:
            log.info(
                f"> Epochs: {e+1}/{epochs} - Train Loss: {train_loss} - Train Acc: {train_acc} - Val Loss: {val_loss} - Val Acc: {val_acc}"
            )

        # Saving infos on a history dict
        for key, value in zip(history, [train_loss, val_loss, train_acc, val_acc]):
            history[key].append(value)

    log.info("Ended training!")

    return history, best_model


def plot_history(history: Dict[str, List[Any]]) -> None:

    # Ploting the Loss and Accuracy Curves
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Loss
    sns.lineplot(data=history["train_loss"], label="Training Loss", ax=ax[0])
    sns.lineplot(data=history["val_loss"], label="Validation Loss", ax=ax[0])
    ax[0].legend(loc="upper right")
    ax[0].set_title("Loss")
    # Accuracy
    sns.lineplot(data=history["train_acc"], label="Training Accuracy", ax=ax[1])
    sns.lineplot(data=history["val_acc"], label="Validation Accuracy", ax=ax[1])
    ax[1].legend(loc="lower right")
    ax[1].set_title("Accuracy")

    plot_filename = _PATH_VISUALIZATION / "model_training.png"
    plt.savefig(plot_filename.resolve())  # CHANGE THIS PATH


if __name__ == "__main__":
    train()

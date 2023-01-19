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
import wandb
from google.cloud import storage
from omegaconf import DictConfig
from src.data.handler import CIFAR10Dataset
from src.models import _PATH_MODELS, _PATH_VISUALIZATION
from src.models.predict_model import validation
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config.yaml", config_path="./")
def train(cfg: DictConfig) -> None:
    """
    Calls the train_models function, with an attached hydra annotation.

    Args:
        cfg: A DictConfig type parsed from .yaml config file via the hydra library.

    Returns:

    """
    train_model(cfg)


def train_model(cfg: DictConfig) -> None:
    """
    A function that trains a given model from the timm-library using the DictConfig class (can be generated via a hydra config file).
    This fuction fetches CIFAR10 dataset from the /data directory, makes a train/validation split and trains the specified model.
    Finally, the function saves the weights and biases in the /models directory.

    Args:
        cfg: A DictConfig type parsed from .yaml config file via the hydra library.

    Returns:

    """
    log.info(f"Running with config: {cfg}")
    # intialize wandb logging to your project
    wandb.init(project="testing cifar10")
    # log all experimental args to wandb
    wandb.config.update(cfg.params)

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

    wandb.watch(model, log_freq=1000)

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
    datetime_str = datetime_now.strftime("%Y-%m-%d-%H-%M-%S")
    state_dict_filename = cfg.params.model + "-" + datetime_str + ".pth"
    state_dict_path = _PATH_MODELS / state_dict_filename
    log.info(f'Saving state dict as "{state_dict_filename}" ')
    torch.save(best_model.state_dict(), state_dict_path.resolve())

    if cfg.params.save_to_cloud:
        bucket_name = "mlops-checkpoints"

        save_to_bucket(str(state_dict_path), bucket_name, state_dict_filename)

    # ploting results
    # plot_history(history)
    torch.load(state_dict_path)
    test_loss, test_accuracy = validation(
        best_model, loss_func, test_dataloader, device,test_flag=1
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
    """
    Carries out a training step for a given model

    Args:
        model: A torch deep learning model
        optimizer: A torch optimizer
        loss_func: A torch loss function
        train_loader: A torch Dataloader set to load training data
        device: Indicator of whether the model should be trained utilising CPU og GPU

    Returns:
        A tuple containing the calculated batch training loss and training accuracy.
    """
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
    """
    A function that start training the model. It compares the best model in the training session with historic
    models and saves the new model configuration, if a better performance is achieved.

    Args:
        model: A torch deep learning model
        optimizer: A torch optimizer
        loss_func: A torch loss function
        scheduler: A torch Scheduler class
        train_loader: A torch Dataloader to load training data
        val_loader: A torch Dataloader to load test data
        epochs: An integer of the number of epochs
        device: Indicator of whether the model should be trained utilising CPU og GPU

    Returns:
        A tuple containing:
        - Historical performance of the model
        - The best performing model configuration from the training
    """
    best_acc = 0
    log.info("Initializing training...")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for e in range(epochs):

        train_loss, train_acc = _training_step(
            model, optimizer, loss_func, train_loader, device
        )
        val_loss, val_acc = validation(model, loss_func, val_loader, device,test_flag=0)
        wandb.log(
            {
                "epoch": e,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )

        scheduler.step()

        if val_acc > best_acc:
            log.info(
                f">> Saving Best Model with Val Acc: Old: {best_acc} | New: {val_acc}"
            )
            best_model = deepcopy(model)
            best_acc = val_acc

        # if (e + 1) % 2 == 0:
        if e >= 0:
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
    print("subplot setup")
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
    plt.show()

    plot_filename = _PATH_VISUALIZATION / "model_training.png"
    print(plot_filename.resolve())
    plt.savefig(plot_filename.resolve())


def save_to_bucket(file_path: str, bucket_name: str, file_name: str) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_path)


if __name__ == "__main__":
    train()

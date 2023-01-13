import sys
import argparse
import os
import copy
import hydra
import timm
import timm.optim

# from timm.data.transforms_factory import create_transform
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from src.data.handler import CIFAR10Dataset
from predict_model import validation_
import seaborn as sns
from torch.optim import lr_scheduler
import numpy as np
import logging

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
pp = os.getcwd()
print("see path: ", pp)


@hydra.main(config_path=pp + "\src\config", config_name="config.yaml")
def main(cfg):
    dataloader_train = CIFAR10Dataset(train=True)
    images = []
    labels = []
    for image, label in dataloader_train:
        images.append(image)
        labels.append(label)
    images = torch.stack(images, dim=0)
    labels = torch.FloatTensor(labels)

    _train, _val = train_test_split(dataloader_train, test_size=0.3, random_state=666)

    train_loader_cifar10 = DataLoader(
        _train, batch_size=cfg.params.batch_size, shuffle=True
    )
    val_loader_cifar10 = DataLoader(
        _val, batch_size=cfg.params.batch_size, shuffle=True
    )

    for i, data in enumerate(train_loader_cifar10, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.reshape(-1, 1)
        inputs = np.array(inputs)
        if i % 500 == 0:
            print("i:", i, " input: ", inputs.shape, " labels: ", labels.shape)
        # TEST DATASET FOR FINAL PREDICTION
        dataloader_test = CIFAR10Dataset(train=False)
        test_loader_cifar10 = DataLoader(
            dataloader_test, batch_size=cfg.params.batch_size, shuffle=True
        )

    # dataloader_train, dataloader_val,images,labels,train_loader_cifar10,val_loader_cifar10 = cifar10()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training Function
    def train_(model, optmizer, loss_func, train_loader, device):
        log.info("Started model training")
        train_loss = 0.0
        train_correct = 0
        size_sampler = len(train_loader.sampler)

        for i, (images, labels) in enumerate(train_loader, 0):

            # Pushing to device (cuda or CPU)
            images, labels = images.to(device), labels.to(device)

            # zeroing gradients
            optmizer.zero_grad()

            # feedforward
            y_hat = model(images)

            # Compute loss
            loss = loss_func(y_hat, labels.long().squeeze())

            # Compute backpropagation
            loss.backward()

            # updating weights
            optmizer.step()

            # loss and correct values compute
            train_loss += loss.item() * images.size(0)
            _, pred = torch.max(y_hat.data, 1)
            train_correct += sum(pred == labels.long().squeeze()).sum().item()

        return np.round(train_loss / size_sampler, 4), np.round(
            train_correct * 100.0 / size_sampler, 3
        )

    # running the model
    def train_model(
        model,
        optmizer,
        loss_func,
        scheduler,
        train_loader,
        val_loader,
        epochs,
        device,
        log=True,
    ):

        best_acc = 0
        print("Initializing Training...")

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for i in range(epochs):

            train_loss, train_acc = train_(
                model, optmizer, loss_func, train_loader, device
            )
            val_loss, val_acc = validation_(
                model, loss_func, val_loader_cifar10, device
            )

            scheduler.step()

            if val_acc > best_acc:
                print(
                    f">> Saving Best Model with Val Acc: Old: {best_acc} | New: {val_acc}"
                )
                best_model = copy.deepcopy(model)
                best_acc = val_acc

            if log and ((i + 1) % 2 == 0):
                print(
                    f"> Epochs: {i+1}/{epochs} - Train Loss: {train_loss} - Train Acc: {train_acc} - Val Loss: {val_loss} - Val Acc: {val_acc}"
                )

            # Saving infos on a history dict
            for key, value in zip(history, [train_loss, val_loss, train_acc, val_acc]):
                history[key].append(value)

        print("...End Training")

        return history, best_model

    def plot_history(history):

        # Ploting the Loss and Accuracy Curves
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

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
        plt.savefig("training.png")  # CHANGE THIS PATH

    # Execute training

    model = timm.create_model(
        cfg.params.model,
        pretrained=True,
        in_chans=cfg.params.in_chans,
        num_classes=cfg.params.num_classes,
    )

    model = model.to(device)

    optmizer = optim.Adam(model.parameters(), lr=cfg.params.learning_rate)
    loss_func = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optmizer, step_size=5, gamma=0.1)

    history, best_model = train_model(
        model=model,
        optmizer=optmizer,
        loss_func=loss_func,
        scheduler=scheduler,
        train_loader=train_loader_cifar10,
        val_loader=val_loader_cifar10,
        epochs=cfg.params.epochs,
        device=device,
    )

    torch.save(best_model.state_dict(), "best_model.pth")

    # ploting results
    plot_history(history)
    torch.load("best_model.pth")
    test_loss, test_accuracy = validation_(
        best_model, loss_func, test_loader_cifar10, device
    )
    print("test_loss: ", test_loss, " test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()

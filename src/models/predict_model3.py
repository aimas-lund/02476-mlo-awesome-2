from make_dataset3 import cifar10
import numpy as np
import torch



def validation_(model, loss_func,val_loader, device):
    dataloader_train, dataloader_val,images,labels,train_loader_cifar10 , val_loader_cifar10 = cifar10()
    val_loss = 0.0
    val_correct = 0
    size_sampler = len(val_loader.sampler)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

            images, labels = images.to(device), labels.to(device)
            y_hat = model(images)
            loss = loss_func(y_hat, labels.long().squeeze())
            
            val_loss +=loss.item() * images.size(0)
            _ , pred = torch.max(y_hat.data, 1)
            val_correct +=(pred == labels.long().squeeze()).sum().item()
        
    return np.round(val_loss/size_sampler,4), np.round(val_correct*100./size_sampler,3)
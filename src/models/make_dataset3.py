# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,ConcatDataset,DataLoader

class confiFile():
    """Configuration class for easy parametrization"""
    
    #Pretrained model with timm
    model = 'resnet10t'
    epochs = 2
    
    in_chans = 3
    num_classes = 10
    learning_rate = 1e-3
    
    val_size = 0.3
    batch_size = 8
    


def cifar10():

    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    os.chdir(r"C:\Users\Antarlina Mukherjee\Documents\Studies\MLOps\Final_project\02476-mlo-awesome-2\src\models")
    pp = os.path.dirname(os.path.dirname(os.getcwd()))   

    DATA_PATH_RAW_TRAIN = pp+r"\data\raw\train"
    DATA_PATH_RAW_VAL = pp+r"\data\raw\val"
    DATA_PATH_PROC_TRAIN = pp+r'\data\processed\train'
    print("data path proc train: ",DATA_PATH_PROC_TRAIN)
    DATA_PATH_PROC_VAL = pp+r'\data\processed\val'

    # Define a transform to normalize the data
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, ), (0.5, ),(0.5,))])
    # define the two dataloaders for the training and validation set 
    CFG = confiFile()
    dataloader_train = CIFAR10(root=DATA_PATH_RAW_TRAIN, download=True,transform=transformer,train=True)
    dataloader_val = CIFAR10(root=DATA_PATH_RAW_VAL,download=True, transform=transformer,train=False)
    
    # save the training set and corresponding labels to tensors in processed directory 
    images = []
    labels = []
    for image,label in dataloader_train:
        #print(y)
        images.append(image)
        labels.append(label)
    images = torch.stack(images, dim=0)
    labels = torch.FloatTensor(labels)
    print("proc train: ",DATA_PATH_PROC_TRAIN+'\images.pt')
    torch.save(images, DATA_PATH_PROC_TRAIN+'\images.pt')
    print("image save complete")
    torch.save(labels, DATA_PATH_PROC_TRAIN+'\labels.pt')
    print("labels save complete")

    _train, _val = train_test_split(dataloader_train, test_size = .3, \
                                                  random_state = 666)

    train_loader_cifar10 = DataLoader(_train, batch_size  = CFG.batch_size,  shuffle = True)
    val_loader_cifar10 = DataLoader(_val, batch_size  = CFG.batch_size,  shuffle = True)
    # for epoch in range(1):
    for i, data in enumerate(train_loader_cifar10, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.reshape(-1,1)
        inputs = np.array(inputs)
        # print("i:",i, " input: ", inputs.shape, " labels: ", labels.shape)
        
    return dataloader_train, dataloader_val,images,labels,train_loader_cifar10,val_loader_cifar10 # train_set, test_set
# dataloader_train, dataloader_val,images, labels= cifar10()
# print('-----------------------------')



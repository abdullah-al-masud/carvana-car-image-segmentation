"""
This script contains necessary functions and classes for modeling purpose.
"""

import numpy as np
import torch
from msdlib import mlutils
import os
from PIL import Image
import torch.nn.functional as F
from unet import UNET


class GlobalAverage(torch.nn.Module):
    """
    This class is a custom pytorch layer which will be used for averaging the channel elements to get final image output at output layer
    
    Inputs:
        :axis: int, dimension of image along which averaging method will be applied. Default is 0
    """
    
    def __init__(self, axis=0):
        super(GlobalAverage, self).__init__()
        self.axis = axis
    
    def forward(self, x):
        return x.mean(axis=self.axis)


def build_model(savepath, modelname, batch_size, epoch, lr, lr_reduce):
    """
    This function creates model architecture and feeds it to torchModel class to get the model ready for training and prediction in easier way.
    
    Inputs:
        :savepath: str, path to the directory where learning curve or evaluation results ill be stored after model training
        :batch_size: int, batch size for model training
        :epoch: int, number of epochs for training
        :lr: float, learning rate of the model during training
        :lr_reduce: float, learning rate reduction factor after each epoch. Should be 0 ~ 1, preferably close to 1.
    """
    
    layers = [
        torch.nn.Conv2d(3, 16, 3, 1, 'same'),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 5, 1, 'same'),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(16, 32, 3, 1, 'same'),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(32, 32, 2, stride=2),
        torch.nn.Conv2d(32, 16, 3, 1, 'same'),
        GlobalAverage(1), # torch.nn.Conv2d(16, 1, 1, 1, 'same'),
        torch.nn.Sigmoid()
    ]

    if 'invision' in modelname:
        model = mlutils.torchModel(layers=layers, model_type='binary-classifier', model_name=modelname, # loss_func=DiceLoss,
                                   savepath=savepath, batch_size=batch_size, epoch=epoch, 
                                   learning_rate=lr, lr_reduce=lr_reduce, interval=10, use_gpu=True, gpu_devices=[2])
    elif 'unet' in modelname:
        unet = UNET(3, 1)
        model = mlutils.torchModel(model=unet, model_type='binary-classifier', loss_func=torch.nn.BCEWithLogitsLoss,
                                   savepath=savepath, batch_size=batch_size, epoch=epoch, tensorboard_path='runs', 
                                   learning_rate=lr, lr_reduce=lr_reduce, interval=20, use_gpu=True, gpu_devices=[0],
                                   model_name=modelname)
    
    return model


class CustomDataSet(torch.utils.data.Dataset):
    """
    This is a customized Data set object which can build a torch.utils.data.Dataset object given the data and labels. 
    This is only usable when we have complete data and labels (data and label lengths must be equal)

    Inputs:
        :data: ideally should be numpy ndarray, pandas DataFrame or torch tensor. Contains feature data tor model training.
               Can be python list or python set too but not much appreciated as it will be mostly used for training pytorch model.
        :label: ideally should be numpy ndarray, pandas Series/DataFrame or torch tensor. Contains true labels tor model training.
                Can be python list or python set too but not much appreciated as it will be mostly used for training pytorch model.
    """

    def __init__(self, image_dir, label_dir, label2image_replacer=['_mask.gif', '.jpg'],
                 dtype=torch.float32, transform=None):

        self.imdir = image_dir
        self.lbdir = label_dir
        self.label_names = np.array(os.listdir(self.lbdir))
        self.replacer = label2image_replacer
        self.datalen = len(self.label_names)
        self.dtype = dtype
        self.transform = transform
        
    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        # data collection and conversion
        name = self.label_names[index]
        data = np.array(Image.open(os.path.join(self.imdir, name.replace(self.replacer[0], self.replacer[1]))))
        label = np.array(Image.open(os.path.join(self.lbdir, name)))
        
        if data.max() > 1:
            data = data.astype(float) / 255.0
        if label.max() > 1:
            label = label.astype(float) / 255.0
        if self.transform is not None:
            data, label = self.transform(data, label)
        
        data = torch.from_numpy(data).view(data.shape[2], data.shape[0], data.shape[1]).to(dtype=self.dtype)
        label = torch.from_numpy(label).to(dtype=self.dtype)
        
        return data, label


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, **kwargs):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

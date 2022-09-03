import os
from model import build_model, CustomDataSet
import torch
from msdlib import mlutils
from inputs import get_params


def main():
    
    params = get_params()
    savedir = params['savedir']
    image_dim = params['image_dim']
    train_path = params['train_path']
    validation_path = params['validation_path']
    batch_size = params['batch_size']
    epoch = params['epoch']
    lr = params['lr']
    lr_reduce = params['lr_reduce']
    modelname = params['modelname']
    
    model = build_model(savedir, modelname, batch_size, epoch, lr, lr_reduce)
    print(model.model)
    
    dataset = CustomDataSet(os.path.join(train_path, 'images'),
                            os.path.join(train_path, 'labels'),
                            label2image_replacer=['_mask.gif', '.jpg'])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = CustomDataSet(os.path.join(validation_path, 'images'),
                            os.path.join(validation_path, 'labels'),
                            label2image_replacer=['_mask.gif', '.jpg'])
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.fit(train_loader=train_loader, val_loader=val_loader, evaluate=False)


if __name__ == "__main__":
    main()

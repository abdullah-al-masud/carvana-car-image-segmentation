import os


def get_params():
    params = dict(
        image_path = 'images',
        label_path = 'masks',
        savedir = 'results',
        image_dim = (512, 512),
        train_path = 'train512',
        validation_path = 'validation512',
        validation_ratio = .15,
        
        batch_size = 16,
        epoch = 100,
        lr = .001,
        lr_reduce = .995,
        
        modelname = 'pytorch-unet'
    )

    return params

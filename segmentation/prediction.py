import argparse
import torch
import numpy as np
import os
from inputs import get_params
from PIL import Image
from model import build_model
from msdlib.mlutils import load_models
from msdlib import msd
import cv2
import matplotlib.pyplot as plt
import pdb



def show_image(image_array, rgb=None, show=True, savepath=None):
    if rgb is not None:
        blank_array = np.zeros((image_array.shape[0], image_array.shape[1], 1)).astype(int)
        if rgb == 'r':
            image_array = np.concatenate([image_array[:, :, np.newaxis], blank_array, blank_array], axis=2)
        elif rgb == 'g':
            image_array = np.concatenate([blank_array, image_array[:, :, np.newaxis], blank_array], axis=2)
        elif rgb == 'b':
            image_array = np.concatenate([blank_array, blank_array, image_array[:, :, np.newaxis]], axis=2)
        else:
            raise ValueError('rgb value "%s" not valid.'%rgb)
    
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap='jet')
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(os.path.join(savepath), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def get_model(model_path, modelname, args, batch_size=1, epoch=1, lr=.0001, lr_reduce=.99, savepath=None):
    
    model = build_model(savepath, modelname, batch_size, epoch, lr, lr_reduce)
    loaded = False
    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            model.model.load_state_dict(torch.load(args.model_path))
            loaded = True
    if not loaded:
        modeldict = {modelname: model.model}
        modeldict = load_models(modeldict, model_path)
        model.model = modeldict[modelname]
    
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', help='path to the input image to be used for prediction')
    parser.add_argument('--label-path', default=None, help='path to the input label to be used for evaluation')
    parser.add_argument('--save-path', default=None,
                        help='path to the output directory where the results and prediction will be saved')
    parser.add_argument('--model-path', default=None,
                        help='path to the trained model')
    args = parser.parse_args()
    return args


def main():
    
    params = get_params()
    image_dim = params['image_dim']
    savedir = params['savedir']
    model_path = params['savedir']
    modelname = params['modelname']
    
    args = get_args()
    image_path = args.image_path
    label_path = args.label_path
    save_path = args.save_path
    
    image = np.array(Image.open(image_path))
    image = torch.from_numpy(cv2.resize(image, (image_dim[0], image_dim[1]), interpolation=cv2.INTER_AREA) / 255.0)
    image = image.view(3, image_dim[0], image_dim[1])
    model = get_model(model_path, modelname, args, batch_size=1, epoch=1, lr=.0001, lr_reduce=.99, savepath=None)
    pred = torch.sigmoid(model.predict(image.unsqueeze(0))).detach().cpu().squeeze().numpy()

    if label_path is not None:
        if os.path.isfile(label_path):
            print('running evaluation for inserted label')
            label = (np.array(Image.open(label_path)) * 255).astype(int)
            show_image(label, savepath=os.path.join(save_path, 'label.png'))
    # pdb.set_trace()
    show_image(pred, savepath=os.path.join(save_path, 'prediction.png'))
    score, confmat = msd.class_result((label / 255).astype(int).ravel(), pred.round().astype(int).ravel(), True)
    print(score)
    print(confmat)


if __name__ == "__main__":
    main()

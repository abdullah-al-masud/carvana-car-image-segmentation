from PIL import Image
import numpy as np
import os
from inputs import get_params
import cv2


def main():
    
    params = get_params()

    images = os.listdir(params['image_path'])
    labels = os.listdir(params['label_path'])
    image_dim = params['image_dim']
    val_ratio = params['validation_ratio']

    index = list(range(len(images)))
    np.random.shuffle(index)
    val_index = index[:int(len(images) * val_ratio)]
    train_index = index[int(len(images) * val_ratio):]
    os.makedirs(os.path.join('train512', 'images'), exist_ok=True)
    os.makedirs(os.path.join('train512', 'labels'), exist_ok=True)
    os.makedirs(os.path.join('validation512', 'images'), exist_ok=True)
    os.makedirs(os.path.join('validation512', 'labels'), exist_ok=True)
    
    for i in val_index:
        if labels[i].replace('_mask.gif', '.jpg') in images:
            label = np.array(Image.open(os.path.join('masks', labels[i])))
            label = (cv2.resize(label, (image_dim[0], image_dim[1]), interpolation=cv2.INTER_AREA) * 255).astype(np.int8)
            Image.fromarray(label).save(os.path.join('validation512', 'labels', labels[i]))

            data = np.array(Image.open(os.path.join('images', labels[i].replace('_mask.gif', '.jpg'))))
            data = cv2.resize(data, (image_dim[0], image_dim[1]), interpolation=cv2.INTER_AREA)
            Image.fromarray(data).save(os.path.join('validation512', 'images', labels[i].replace('_mask.gif', '.jpg')))

    for i in train_index:
        if labels[i].replace('_mask.gif', '.jpg') in images:
            label = np.array(Image.open(os.path.join('masks', labels[i])))
            label = (cv2.resize(label, (image_dim[0], image_dim[1]), interpolation=cv2.INTER_AREA) * 255).astype(np.int8)
            Image.fromarray(label).save(os.path.join('train512', 'labels', labels[i]))

            data = np.array(Image.open(os.path.join('images', labels[i].replace('_mask.gif', '.jpg'))))
            data = cv2.resize(data, (image_dim[0], image_dim[1]), interpolation=cv2.INTER_AREA)
            Image.fromarray(data).save(os.path.join('train512', 'images', labels[i].replace('_mask.gif', '.jpg')))


if __name__ == "__main__":
    main()
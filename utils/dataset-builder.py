# Data Augmentation

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = ab[0]['train']
    train_y = ab[1]['train']

    test_x = ab[0]['test']
    test_y = ab[1]['test']
    
    val_x = ab[0]['val']
    val_y = ab[1]['val']


    return (train_x, train_y), (test_x, test_y), (val_x, val_y)

def augment_data(images, masks, save_path, augment=True):
    size = (480, 480)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            aug = A.Transpose(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]
            
            aug = A.Affine(scale=(1.5, 2.0),  rotate=(-45, 45),  shear=(-10, 10),   p=1)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]
            
            aug = A.ShiftScaleRotate(p = 1)
            augmented = aug(image=x, mask=y)
            x6 = augmented["image"]
            y6 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5, x6]
            Y = [y, y1, y2, y3, y4, y5, y6]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "/kaggle/input/drive2004/DRIVE/"
    (train_x, train_y), (test_x, test_y), (val_x, val_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    print(f"val: {len(val_x)} - {len(val_y)}")

    """ Create directories to save the augmented data """
    create_dir("drive/train/image/")
    create_dir("drive/train/mask/")
    create_dir("drive/test/image/")
    create_dir("drive/test/mask/")
    create_dir("drive/val/image/")
    create_dir("drive/val/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "drive/train/", augment=True)
    augment_data(test_x, test_y, "drive/test/", augment=False)
    augment_data(val_x, val_y, "drive/val/", augment=False)
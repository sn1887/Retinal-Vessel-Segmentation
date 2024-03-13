# Data Augmentation

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import torch
from torch.utils.data import Dataset

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
    
    
#-------------------------------------------------------------------------------------------


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
    
    
#-------------------------------------------------------------------------------------------    
    
if __name__ == "__main__":
    """ Seeding """
    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("/kaggle/working/drive/train/image/*"))
    train_y = sorted(glob("/kaggle/working/drive/train/mask/*"))

    valid_x = sorted(glob("/kaggle/working/drive/val/image/*"))
    valid_y = sorted(glob("/kaggle/working/drive/val/mask/*"))
    
    test_x = sorted(glob("/kaggle/working/drive/test/image/*"))
    test_y = sorted(glob("/kaggle/working/drive/test/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 480
    W = 480
    size = (H, W)
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)
    test_dataset = DriveDataset(test_x, test_y)
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Rotate,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout,
    CoarseDropout, ShiftScaleRotate, CenterCrop, Resize)
from albumentations.pytorch import ToTensorV2


class VinData(Dataset):
    def __init__(self, df, config=None, transforms=None):
        self.df = df
        self.config = config
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.config.TRAIN_PATH, self.df.loc[idx,'image_id']+'.png')
        img = cv2.imread(img_path, 0)
        img = np.stack([img,img,img], axis=2)
        labels = self.df.loc[idx, 'label']

        if self.transform:
            img = self.transform(image=img)['image'] # return (C x H x W)

        return img, labels

# Augmentations
def get_train_transforms(config):
    return Compose([
            RandomResizedCrop(config.image_size, config.image_size),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Transpose(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.4),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.4),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms(config):
    return Compose([
            CenterCrop(config.image_size, config.image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

# Prepare dataloaders
def prepare_loader(train_df, valid_df, config):
    train_ds = VinData(train_df, config, transforms=get_train_transforms(config))
    valid_ds = VinData(valid_df, config, transforms=get_valid_transforms(config))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2)

    val_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=False)

    return train_loader, val_loader

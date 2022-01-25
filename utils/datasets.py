import torch
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import os

class SegDataset(BaseDataset):    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('.jpg', '.png')) for image_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i], 0)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask > 0)]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class ClsDataset(BaseDataset):

    def __init__(self, df, image_dir, mask_dir, augmentation=None, preprocessing=None):

        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(os.path.join(self.image_dir, row.StudyInstanceUID + '.jpg'), 0)
        mask = cv2.imread(os.path.join(self.mask_dir, row.StudyInstanceUID + '.jpg.png'), 0)

        masks = [(mask > 0)]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = np.concatenate([image, mask], 0)
        
        label = row[[
            'ETT - Abnormal',
            'ETT - Borderline',
            'ETT - Normal'
        ]].values.astype(float)

        return image, torch.tensor(label).float()
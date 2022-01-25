import numpy as np
import albumentations
import torch

transforms_train = albumentations.Compose([
    albumentations.Resize(512, 512),                                    
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightness(limit=0.1, p=0.75),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.75),
    albumentations.Cutout(max_h_size=int(512 * 0.3), max_w_size=int(512 * 0.3), num_holes=1, p=0.75),
])
transforms_val = albumentations.Compose([
    albumentations.Resize(512, 512),
])

def normalization1(x, **kwargs):
    return (x - np.mean(x)) / np.std(x)

def to_tensor(x, **kwargs):
    x = torch.from_numpy(x).float()
    return x

def to_tensor2(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')  #RGB

def add_dimention(x, **kwargs):
    return np.expand_dims(x, axis=0)

def get_preprocessing():
    _transform = [
        albumentations.Lambda(image=normalization1),
        albumentations.Lambda(image=add_dimention),
        albumentations.Lambda(image=to_tensor, mask=to_tensor2),
    ]
    return albumentations.Compose(_transform)
import imp
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from utils.preprocessing import *

model = torch.load()

model.eval()
image_path = 'D:/Kaggle/ETT_image'
save_path = 'D:/Kaggle/output_mask'
with torch.no_grad():
    for image_id in tqdm(os.listdir(image_path)):
        image = cv2.imread(os.path.join(image_path, image_id), 0)
        image = cv2.resize(image, (512, 512))
        image = normalization1(image)
        image = add_dimention(image)
        image = to_tensor(image)
        image = image.unsqueeze(0).cuda()

        output = model(image).squeeze().cpu().numpy()
        
        output = np.where(output > 0.5, 255, 0)

        cv2.imwrite(os.path.join(save_path, image_id + '.png'), output)
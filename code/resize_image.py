import torch
from torchvision.io import read_image
from torchvision import transforms
from torchvision.utils import save_image

import os
import numpy as np

import support_victor_machine as supp

resizer = transforms.Compose([transforms.Resize((128, 96))])

base_path = '../images/'

image_directories = os.listdir(base_path)

print(image_directories)

for folder in image_directories:
    path = base_path+folder
    folder_files = os.listdir(path)
    os.mkdir('../resized_128/'+folder)
    print('folder '+folder+' created')
    
    for file in folder_files:
        file_path = path+'/'+file
        
        img = read_image(file_path)
        
        if img.shape[1] < img.shape[2]:
            img = torch.transpose(img, 1,2)
        
        img = resizer(img)
        img = torch.div(img,255)
        
        save_image(img, '../resized_128/'+folder+'/'+file)
    
import random
import os
import numpy as np

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

import transforms


iheight, iwidth = 480, 640 # raw image size

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

class NyuDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        
        if train:
            self.transform = self.train_transform
        else:
            self.transform = self.val_transform
            
        self.filenames = [name for name in os.listdir(self.root)] 
        self.output_size = (224, 224)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        rgb, depth = h5_loader(os.path.join(self.root, self.filenames[idx]))
        rgb, depth = self.transform(rgb, depth)
        
        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(rgb)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth)
        depth_tensor = depth_tensor.unsqueeze(0)
        
        return input_tensor, depth_tensor 
        
    
    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((228, 304)),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        #rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(250 / iheight),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np
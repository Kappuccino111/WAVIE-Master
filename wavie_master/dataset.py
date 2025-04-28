from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from utils import list_images

# Real image = 0
# Fake image = 1
class DeepfakeDataset(Dataset):
    def __init__(self, split, realLoc, fakeLoc,seed=42, transforms=None ):
        self.realLoc = realLoc
        self.realImages = list_images(realLoc,0)
        self.fakeLoc = fakeLoc
        self.fakeImages = list_images(fakeLoc,1)
        self.images=self.realImages+self.fakeImages
        random.seed(seed)
        random.shuffle(self.images)
        start=0, end=1
        if split=="train":
            end=0.6
        elif split=="val":
            start=0.6
            end=0.8
        else:
            start=0.8
            
        total=len(self.images)
        self.transforms = transforms
        self.images=self.images[int(total*start):int(total*end)]

    def __len__(self):
        return len(os.listdir(self.datasetLoc))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return [image, label]
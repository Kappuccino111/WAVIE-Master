from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from utils import list_images, readImages

# Real image = 0
# Fake image = 1
class DeepfakeDataset(Dataset):


    def __init__(self, split, realLoc:str, fakeLoc:str,seed=42, transforms=None ):
        self.realLoc = realLoc
        self.realImages = list_images(realLoc,0)
        self.fakeLoc = fakeLoc
        self.fakeImages = list_images(fakeLoc,1)
        self.images=self.realImages+self.fakeImages
        random.seed(seed)
        random.shuffle(self.images)
        start=0, end=1
        if split=="train":
            start=0
            end=0.8
        elif split=="val":
            start=0.8
            end=1
        elif split=="test":
            start=0
            end=1

        total=len(self.images)
        self.transforms = transforms
        self.images=self.images[int(total*start):int(total*end)]

    def __init__(self, split, seed=42, transforms=None):
        realDirectories = []
        with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/realRatio.txt","r") as f:
            for i in f.readlines():
                if len(i.split(" "))==1:
                    realDirectories.append(tuple(i,1))
                else:
                    realDirectories.append(tuple(i.split(" ")))

        fakeDirectories=[]
        with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/fakeRatio.txt","r") as f:
            for i in f.readlines():
                if len(i.split(" "))==1:
                    fakeDirectories.append(tuple(i,1))
                else:
                    fakeDirectories.append(tuple(i.split(" ")))
        

        random.seed(seed)

        realImages = []
        for i,j in realDirectories:
            ratio=float(j)
            curr=[]
            with open(f"{i}/info$na.txt", "r") as f:
                for line in f.readlines():
                    curr.append(line)
            num=None
            if ratio<=1:
                num=int(ratio*len(curr))
            else:
                num=int(ratio)
            sampled = random.sample(curr, num)
            realImages=realImages+sampled
        

        fakeImages = []
        for i,j in fakeDirectories:
            ratio=float(j)
            curr=[]
            with open(f"{i}/info$na.txt", "r") as f:
                for line in f.readlines():
                    curr.append(line)
            num=None
            if ratio<=1:
                num=int(ratio*len(curr))
            else:
                num=int(ratio)
            sampled = random.sample(curr, num)
            fakeImages=fakeImages+sampled

        realImages = [(i,0) for i in realImages]
        fakeImages = [(i,1) for i in fakeImages]

        self.realImages=realImages
        self.fakeImages=fakeImages
        

        self.images=self.realImages+self.fakeImages

        random.shuffle(self.images)
        start=0, end=1
        if split=="train":
            start=0
            end=0.8
        elif split=="val":
            start=0.8
            end=1
        elif split=="test":
            start=0
            end=1

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
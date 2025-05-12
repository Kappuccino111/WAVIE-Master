from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from .utils import list_images, readImages

# Real image = 0
# Fake image = 1
class DeepfakeDataset(Dataset):

    def __init__(self, split, realLoc=None, fakeLoc=None, seed=42, transforms=None):
        import os, random
        from .utils import list_images

        self.transforms = transforms
        random.seed(seed)

        # --- gather (path, label) pairs ---
        if realLoc is not None and fakeLoc is not None:
            # fixed-folder mode
            real_images = list_images(realLoc, 0)
            fake_images = list_images(fakeLoc, 1)
        else:
            # ratio-file mode
            real_images, fake_images = [], []
            # load directories + ratios
            base = "/home/teaching/Desktop/WAVIE-Master/wavie_master"
            real_dirs = []
            with open(os.path.join(base, "realRatio.txt"), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    real_dirs.append((parts[0], float(parts[1]) if len(parts) > 1 else 1.0))
            fake_dirs = []
            with open(os.path.join(base, "fakeRatio.txt"), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    fake_dirs.append((parts[0], float(parts[1]) if len(parts) > 1 else 1.0))

            # sample real
            for directory, ratio in real_dirs:
                infos = [ln.strip() for ln in open(os.path.join(directory, "info$na.txt")) if ln.strip()]
                n = int(ratio * len(infos)) if ratio <= 1 else int(ratio)
                sampled = random.sample(infos, n)
                real_images += [(os.path.join(directory, fn), 0) for fn in sampled]

            # sample fake
            for directory, ratio in fake_dirs:
                infos = [ln.strip() for ln in open(os.path.join(directory, "info$na.txt")) if ln.strip()]
                n = int(ratio * len(infos)) if ratio <= 1 else int(ratio)
                sampled = random.sample(infos, n)
                fake_images += [(os.path.join(directory, fn), 1) for fn in sampled]

        # combine and shuffle
        self.images = real_images + fake_images
        random.shuffle(self.images)

        # --- split ---
        total = len(self.images)
        if split == "train":
            end = int(0.8 * total)
            self.images = self.images[:end]
        elif split == "val":
            start = int(0.8 * total)
            self.images = self.images[start:]
        # else "test": keep all

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return [image, label]
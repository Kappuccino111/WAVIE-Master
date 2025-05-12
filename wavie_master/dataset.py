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


    # def __init__(self, split, realLoc:str, fakeLoc:str,seed=42, transforms=None ):
    #     self.realLoc = realLoc
    #     self.realImages = list_images(realLoc,0)
    #     self.fakeLoc = fakeLoc
    #     self.fakeImages = list_images(fakeLoc,1)
    #     self.images=self.realImages+self.fakeImages
    #     random.seed(seed)
    #     random.shuffle(self.images)
    #     start=0
    #     end=1
    #     if split=="train":
    #         start=0
    #         end=0.8
    #     elif split=="val":
    #         start=0.8
    #         end=1
    #     elif split=="test":
    #         start=0
    #         end=1

    #     total=len(self.images)
    #     self.transforms = transforms
    #     self.images=self.images[int(total*start):int(total*end)]

    # def __init__(self, split, seed=42, transforms=None):
    #     realDirectories = []
    #     with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/realRatio.txt", "r") as f:
    #         for line in f:
    #             parts = line.strip().split()
    #             # either ("path", "ratio") or ("path",) → make it ("path", "1")
    #             realDirectories.append((parts[0], parts[1] if len(parts) > 1 else "1"))

    #     fakeDirectories = []
    #     with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/fakeRatio.txt", "r") as f:
    #         for line in f:
    #             parts = line.strip().split()
    #             fakeDirectories.append((parts[0], parts[1] if len(parts) > 1 else "1"))

    #     # --- build realImages list of (full_path, 0) ---
    #     realImages = []
    #     for dirpath, ratio_str in realDirectories:
    #         ratio = float(ratio_str)
    #         # read filenames, strip whitespace/newlines
    #         with open(os.path.join(dirpath, "info$na.txt"), "r") as f_img:
    #             all_files = [ln.strip() for ln in f_img if ln.strip()]
    #         # decide how many to sample
    #         n = int(ratio * len(all_files)) if ratio <= 1 else int(ratio)
    #         sampled = random.sample(all_files, n)
    #         # prepend directory and label
    #         realImages += [
    #             (os.path.join(dirpath, fname), 0)
    #             for fname in sampled
    #         ]

    #     # --- build fakeImages list of (full_path, 1) ---
    #     fakeImages = []
    #     for dirpath, ratio_str in fakeDirectories:
    #         ratio = float(ratio_str)
    #         with open(os.path.join(dirpath, "info$na.txt"), "r") as f_img:
    #             all_files = [ln.strip() for ln in f_img if ln.strip()]
    #         n = int(ratio * len(all_files)) if ratio <= 1 else int(ratio)
    #         sampled = random.sample(all_files, n)
    #         fakeImages += [
    #             (os.path.join(dirpath, fname), 1)
    #             for fname in sampled
    #         ]

    #     # combine, shuffle, and split
    #     self.images = realImages + fakeImages
    #     random.seed(seed)
    #     random.shuffle(self.images)

    #     # train/val/test split
    #     total = len(self.images)
    #     if split == "train":
    #         self.images = self.images[: int(0.8 * total)]
    #     elif split == "val":
    #         self.images = self.images[int(0.8 * total) :]
    #     # else split=="test" → keep all

    #     self.transforms = transforms

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
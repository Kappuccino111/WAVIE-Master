#!/usr/bin/env python3
import sys
from pathlib import Path
import os
import torch
from io import BytesIO
from torchvision import transforms
import random
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from PIL import Image
import cv2

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)

def data_augment(img):
    img = np.array(img)

    if random.random() < 0.5:
        sig = sample_continuous([0.0, 3.0])
        gaussian_blur(img, sig)

    if random.random() < 0.5:
        method = sample_discrete(["cv2", "pil"])
        qual = sample_discrete([30, 100])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)

def get_transforms():
    transforms_train = transforms.Compose(
        [
            transforms.Lambda(lambda img: data_augment(img)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    transforms_test_1 = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    transforms_test_2 = transforms.Compose(
        [
            transforms.TenCrop(224),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.PILToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return transforms_train, transforms_test_1, transforms_test_2



def list_images(base_folder: Path,i):
    """
    Recursively find image files under base_folder and print their paths
    relative to base_folder.
    """
    # Define the set of image extensions to look for
    img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    lst=[]
    # Walk the tree
    for p in base_folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in img_exts:
            # Print path relative to the base folder

            lst.append((os.path.join(os.getcwd(), base_folder, p.relative_to(base_folder)),i))
    return lst

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model and optimizer state to a checkpoint file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path, device):
    """
    Load model (and optimizer) state from checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint.get('epoch', None)

def compute_accuracy(logits, labels):
    """
    Compute binary classification accuracy given logits and true labels.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    correct = (preds.view(-1) == labels.view(-1)).sum().item()
    total = labels.numel()
    return correct / total if total > 0 else 0.0

#!/usr/bin/env python3
import sys
from pathlib import Path
import os
import torch

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

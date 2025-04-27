#!/usr/bin/env python3
import sys
from pathlib import Path
import os
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


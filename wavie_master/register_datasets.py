import os
import sys
from utils import list_images
from pathlib import Path

def list_images(base_folder: Path):
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
            if "/home/teaching" in str(base_folder):
                lst.append(os.path.join(base_folder, p.relative_to(base_folder)))
            else:
                lst.append(os.path.join(os.getcwd(), base_folder, p.relative_to(base_folder)))
    return lst


def saveRealData(data, base_folder):
    if "/home/teaching" in str(base_folder):
        pass
    else:
        base_folder=os.path.join(os.getcwd(), base_folder)
    
    with open(os.path.join(base_folder, "info$na.txt"),"w") as f:
        for i in data:
            f.write(f"{i}\n")
    
    with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/realRatio.txt", "r") as f:
        file_data = f.read()
    if str(base_folder) not in file_data:
        with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/realRatio.txt", "a") as f:
            f.write(f"{base_folder}\n")

def saveFakeData(data, base_folder):
    if "/home/teaching" in str(base_folder):
        pass
    else:
        base_folder=os.path.join(os.getcwd(), base_folder)
    
    with open(os.path.join(base_folder, "info$na.txt"),"w") as f:
        f.writelines(data)
    
    with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/fakeRatio.txt", "r") as f:
        file_data = f.read()
    if str(base_folder) not in file_data:
        with open("/home/teaching/Desktop/WAVIE-Master/wavie_master/fakeRatio.txt", "a") as f:
            f.write(f"{base_folder}\n")
    
if __name__=="__main__":
    loc = Path(sys.argv[1])
    if "/home/teaching" in str(loc):
        pass
    else:
        loc = Path(os.getcwd()) / loc
    image_type = sys.argv[2]
    images = list_images(loc)

    if str(image_type)=="0":
        saveRealData(images, loc)
    elif str(image_type)=="1":
        saveFakeData(images, loc)

    

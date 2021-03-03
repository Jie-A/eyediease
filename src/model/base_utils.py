import re
from pathlib import Path
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List
from PIL import Image
from skimage.io import imread as mask_read
from catalyst.contrib.utils.cv import image as cata_image


lesion_paths = {
    'MA': '1. Microaneurysms',
    'EX': '3. Hard Exudates',
    'HE': '2. Haemorrhages',
    'SE': '4. Soft Exudates'
}


def get_datapath(img_path: Path, mask_path: Path, lesion_type: str = 'EX'):
    lesion_path = lesion_paths[lesion_type]
    img_posfix = '.jpg'
    mask_posfix = '_' + lesion_type + '.tif'
    mask_names = os.listdir(os.path.join(mask_path, lesion_path))

    mask_ids = list(map(lambda mask: re.sub(
        mask_posfix, '', mask), mask_names))

    restored_name = list(map(lambda x: x + img_posfix, mask_ids))

    full_img_paths = list(
        map(lambda x: Path(os.path.join(img_path, x)), restored_name))
    full_mask_paths = list(
        map(lambda x: Path(os.path.join(mask_path, lesion_path, x)), mask_names))

    print('[INFO] full img paths', len(full_img_paths))
    print('[INFO] full_mask_paths', len(full_mask_paths))

    return sorted(full_img_paths), sorted(full_mask_paths)


def show_examples(name: str, image: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")


def show(index: int, images: List[Path], masks: List[Path], transforms=None) -> None:
    image_path = images[index]
    name = image_path.name

    image = cata_image.imread(image_path)
    mask = mask_read(masks[index])

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask)


def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)


def save_output(pred_masks: np.ndarray, out_path: Path):
    # Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / pred_masks.max() *
                (pred_masks - pred_masks.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(out_path)
    print(f'[INFO] saved {out_path.name} to disk')
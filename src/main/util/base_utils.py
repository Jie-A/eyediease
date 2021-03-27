import re
from pathlib import Path
import collections
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Callable
from PIL import Image
from skimage.io import imread as mask_read
from catalyst.contrib.utils.cv import image as cata_image
from catalyst.utils.distributed import (
    get_distributed_env,
    get_distributed_params,
)
import warnings
import subprocess
import sys
import torch

from prettytable import PrettyTable


Lesion = collections.namedtuple('Lesion', ['dir_name', 'project_name'])

lesion_dict = {
    'MA': Lesion( dir_name='1. Microaneurysms', project_name='MicroaneurysmsSegmentation'),
    'EX': Lesion(dir_name='3. Hard Exudates', project_name='HardExudatesSegmentation'),
    'HE': Lesion(dir_name='2. Haemorrhages',
           project_name='HaemorrhageSegmentation'),
    'SE': Lesion(dir_name='4. Soft Exudates',
           project_name='SoftExudatesSegmentation')
}



def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img
    

def get_datapath(img_path: Path, mask_path: Path, lesion_type: str = 'EX'):
    lesion_path = lesion_dict[lesion_type].dir_name
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
    mask = mask_read(masks[index]).astype(np.float32)

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
    rescaled = (255.0 / (pred_masks.max() + np.finfo(float).eps) *
                (pred_masks - pred_masks.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(out_path)
    print(f'[INFO] saved {out_path.name} to disk')


def log_pretty_table(col_names, row_data):
    x = PrettyTable()

    x.field_names = col_names
    for row in row_data:
        x.add_row(row)

    print(x)

def distributed_cmd_run(
    worker_fn: Callable, distributed: bool = True, *args, **kwargs
) -> None:
    """
    Distributed run
    Args:
        worker_fn: worker fn to run in distributed mode
        distributed: distributed flag
        args: additional parameters for worker_fn
        kwargs: additional key-value parameters for worker_fn
    """
    distributed_params = get_distributed_params()
    local_rank = distributed_params["local_rank"]
    world_size = distributed_params["world_size"]

    if distributed and torch.distributed.is_initialized():
        warnings.warn(
            "Looks like you are trying to call distributed setup twice, "
            "switching to normal run for correct distributed training."
        )

    if (
        not distributed
        or torch.distributed.is_initialized()
        or world_size <= 1
    ):
        worker_fn(*args, **kwargs)
    elif local_rank is not None:
        torch.cuda.set_device(int(local_rank))

        torch.distributed.init_process_group(
            backend="gloo", init_method="env://"
        )
        worker_fn(*args, **kwargs)
    else:
        workers = []
        try:
            for local_rank in range(torch.cuda.device_count()):
                rank = distributed_params["start_rank"] + local_rank
                env = get_distributed_env(local_rank, rank, world_size)
                cmd = [sys.executable] + sys.argv.copy()
                workers.append(subprocess.Popen(cmd, env=env))
            for worker in workers:
                worker.wait()
        finally:
            for worker in workers:
                worker.kill()

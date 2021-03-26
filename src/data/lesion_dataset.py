import re
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from typing import List
from pathlib import Path
from pytorch_toolbelt.utils import fs, image_to_tensor
from skimage.io import imread as mask_read
from catalyst.contrib.utils.cv import image as cata_image
import numpy as np
from iglovikov_helper_functions.utils.image_utils import pad
from collections import OrderedDict
import cv2


__all__ = ['CLASS_NAMES', 'CLASS_COLORS', 'OneLesionSegmentation',
           'MultiLesionSegmentation', 'TestSegmentation']


CLASS_NAMES = [
    'MA',
    'EX',
    'HE',
    'SE'
]

CLASS_COLORS = [
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0)
]

lesion_paths = {
    'MA': '1. Microaneurysms',
    'EX': '3. Hard Exudates',
    'HE': '2. Haemorrhages',
    'SE': '4. Soft Exudates'
}


class OneLesionSegmentation(Dataset):
    def __init__(self, images: List[Path], masks: List[Path] = None, transform=None, preprocessing_fn=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]

        image = cata_image.imread(image_path)
        mask = mask_read(self.masks[index]).astype(np.float32)

        if self.transform is not None:
            results = self.transform(image=image, mask=mask)
            image, mask = results['image'], results['mask']

        if self.preprocessing_fn is not None:
            result = self.preprocessing_fn(image=image)
            image = result['image']

        image = image_to_tensor(image).float()
        mask = image_to_tensor(mask, dummy_channels_dim=True).float()
        image_id = fs.id_from_fname(image_path)

        return {
            'image': image,
            'mask': mask,
            'image_id': image_id
        }


class MultiLesionSegmentation(Dataset):
    def __init__(self, images: List[Path], mask_dir: str, transform=None, factor=None, preprocessing_fn=None):
        self.images = images
        self.dir = mask_dir
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = cata_image.imread(image_path)

        masks = []
        for clss in CLASS_NAMES:
            mask_name = re.sub('.jpg', '_' + clss + '.tif', image_path.name)
            path = os.path.join(self.dir, lesion_paths[clss], mask_name)
            if os.path.exists(path):
                mask = mask_read(path).astype(np.float32)
                masks.append(mask)

        mask = np.stack(masks, axis=-1).astype(np.float32)

        if self.transform is not None:
            results = self.transform(image=image, mask=mask)
            image, mask = results['image'], results['mask']

        if self.preprocessing_fn is not None:
            result = self.preprocessing_fn(image=image)
            image = result['image']

        image = image_to_tensor(image).float()
        mask = image_to_tensor(mask, dummy_channels_dim=False).float()
        image_id = fs.id_from_fname(image_path)

        return {
            'image': image,
            'mask': mask,
            'image_id': image_id
        }


class TestSegmentation(Dataset):
    def __init__(self, images: List[Path], masks: List[Path] = None, transform=None, factor=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.factor = factor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]

        result = OrderedDict()
        image = cata_image.imread(image_path)
        result['image'] = image
        if self.masks is not None:
            mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
            result['mask'] = mask

        if self.transform is not None:
            transformed = self.transform(**result)
            result['image'] = transformed['image']
            if self.masks is not None:
                result['mask'] = transformed['mask']

        result['filename'] = image_path.name
        if self.factor is not None:
            normalized_image, pads = pad(image, factor=self.factor)
            result['pad'] = np.array(pads)
            result['image'] = normalized_image

        return result


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import sys
    sys.path.append('..')

    from main import config
    from base_utils import minmax_normalize

    configs = config.BaseConfig.get_all_attributes()

    TRAIN_IMG_DIRS = Path(configs['train_img_path'])
    TRAIN_MASK_DIRS = Path(configs['train_mask_path'])

    image_augmenter = None
    dataset = MultiLesionSegmentation(
        images=list(TRAIN_IMG_DIRS.glob('*.jpg')),
        mask_dir=TRAIN_MASK_DIRS,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=default_collate
    )
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels, _ = batched
        if i == 0:
            fig, axes = plt.subplots(8, 2, figsize=(20, 48))
            plt.tight_layout()
            for j in range(8):
                axes[j][0].imshow(minmax_normalize(
                    images[j], norm_range=(0, 1)))
                axes[j][1].imshow(labels[j])
                axes[j][0].set_xticks([])
                axes[j][0].set_yticks([])
                axes[j][1].set_xticks([])
                axes[j][1].set_yticks([])
            plt.savefig('sample/multi_lesion.png')
            plt.close()
        break

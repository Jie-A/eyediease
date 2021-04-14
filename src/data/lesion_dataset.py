import re
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from typing import List
from pathlib import Path
from pytorch_toolbelt.utils import fs, image_to_tensor
from catalyst.contrib.utils.cv import image as cata_image
import numpy as np
from iglovikov_helper_functions.utils.image_utils import pad
from collections import OrderedDict
import cv2
from PIL import Image
from tqdm.auto import tqdm

import rasterio
from rasterio.windows import Window

import sys
sys.path.append('../main/')
from util import make_grid

__all__ = ['CLASS_NAMES', 'CLASS_COLORS', 'OneLesionSegmentation',
           'MultiLesionSegmentation', 'TestSegmentation']


CLASS_NAMES = [
    'MA',
    'EX',
    'HE',
    'SE',
    'OD'
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
    'SE': '4. Soft Exudates',
    'OD': '5. Optic Disc'
}

class OneLesionSegmentation(Dataset):
    def __init__(self, images: List[Path], masks: List[Path] = None, transform=None, preprocessing_fn=None, data_type = 'all'):
        self.images = images
        self.mask_paths = masks
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        self.mode = data_type
        self.len = len(images)
        if data_type == 'tile':
            self.window = 512
            self.overlap = 32
            self.threshold = 10
            self.identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
            self.x, self.y, self.tile_id = [], [], []
            self.build_slide()
            self.len = len(self.x)

    def __len__(self):
        return self.len

    def build_slide(self):
        self.masks = []
        self.files = []
        self.slices = []
        for i, img_path in enumerate(self.images):
            self.files.append(img_path)
            
            with rasterio.open(img_path, transform = self.identity) as dataset:
                mask = Image.open(self.mask_paths[i])
                mask = np.asarray(mask).astype(np.float32)
                self.masks.append(mask)
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)
                
                for j, slc in tqdm(enumerate(slices)):
                    x1,x2,y1,y2 = slc
                    if self.masks[-1][x1:x2,y1:y2].sum() > self.threshold:
                        self.slices.append([i,x1,x2,y1,y2])
                        
                        image = dataset.read([1,2,3],
                            window=Window.from_slices((x1,x2),(y1,y2)))
                        
                        image = np.moveaxis(image, 0, -1)
                        self.x.append(image)
                        self.y.append(self.masks[-1][x1:x2,y1:y2])
                        self.tile_id.append(img_path.name + '_tile_' + str(j))

    def __getitem__(self, index: int) -> dict:
        if self.mode == 'all':
            image_path = self.images[index]
            image = cata_image.imread(image_path)
            mask = Image.open(self.mask_paths[index])
            mask = np.asarray(mask).astype(np.float32)
            image_id = fs.id_from_fname(image_path)
        else:
            image, mask = self.x[index], self.y[index]
            image_id = self.tile_id[index]

        if self.transform is not None:
            results = self.transform(image=image, mask=mask)
            image, mask = results['image'], results['mask']

        if self.preprocessing_fn is not None:
            result = self.preprocessing_fn(image=image)
            image = result['image']

        image = image_to_tensor(image).float()
        mask = image_to_tensor(mask, dummy_channels_dim=True).float()

        assert mask.shape[:1] == torch.Size([1]) and len(mask.shape) == 3, f'Mask shape is {mask.shape}'
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
            image = transformed['image']
            image = image.float()
            result['image'] = image
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

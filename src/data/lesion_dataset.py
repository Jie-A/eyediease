from os import name
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
from pytorch_toolbelt.utils import fs, image_to_tensor
from skimage.io import imread as mask_read
from catalyst.contrib.utils.cv import image as cata_image
import numpy as np
from iglovikov_helper_functions.utils.image_utils import pad
from collections import OrderedDict
import cv2

__all__ = ['OneLesionSegmentation',
           'MultiLesionSegmentation', 'TestSegmentation']


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
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass


class TestSegmentation(Dataset):
    def __init__(self, images: List[Path], masks: List[Path]=None, transform=None, factor=None):
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

from torch.utils.data import Dataset
from typing import List
from pathlib import Path
from pytorch_toolbelt.utils import fs, image_to_tensor
from skimage.io import imread as mask_read
from catalyst.contrib.utils.cv import image as cata_image
import numpy as np

__all__ = ['OneLesionSegmentation', 'MultiLesionSegmentation', 'TestSegmentation']

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
    def __init__(self, images: List[Path], transform=None, preprocessing_fn=None):
        self.images = images
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]

        image = cata_image.imread(image_path)

        if self.transform is not None:
            results = self.transform(image=image)
            image= results['image']

        if self.preprocessing_fn is not None:
            result = self.preprocessing_fn(image=image)
            image = result['image']

        image = image_to_tensor(image).float()
        filename = image_path.name

        return {
            'image': image,
            'filename': filename
        }

import re
import os
from torch.utils.data import Dataset, DataLoader
from typing import List
from pathlib import Path
from pytorch_toolbelt.utils import fs, image_to_tensor
import numpy as np
from iglovikov_helper_functions.utils.image_utils import pad
from collections import OrderedDict
from PIL import Image
from tqdm.auto import tqdm


__all__ = ['CLASS_NAMES', 'CLASS_COLORS', 'MultiLesionSegmentation', 'TestSegmentation']


CLASS_NAMES = [
    'MA',
    'EX',
    'HE',
    'SE',
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
}

lesion = OrderedDict({
    '3. Hard Exudates':'EX',
    '2. Haemorrhages': 'HE',
    '4. Soft Exudates': 'SE',
    '1. Microaneurysms': 'MA'
})

# VesselSegmentation = OneLesionSegmentation

class MultiLesionSegmentation(Dataset):
    def __init__(self, images: List[Path], is_gray: bool, mask_dir: str, transform=None, preprocessing_fn=None):
        self.images = images
        self.is_gray = is_gray
        self.dir = mask_dir
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        img_name = image_path.name
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image).astype('uint8')
        masks = {}
        for c, ds, _ in os.walk(str(self.dir)):
            for d in ds:
                d = os.path.join(c, d)
                for f in os.listdir(d):
                    f_name = f[:8]
                    img_name = img_name.split('.')[0][:8]
                    if f_name == img_name:
                        mask_type = d.split('/')[-1]
                        if mask_type in list(lesion.keys()):
                            masks[lesion[mask_type]] = os.path.join(d, f)
                            break
            
        total_masks = []
        # print(masks)
        for k in ['EX', 'HE', 'SE', 'MA']:
            if masks.get(k, None) is not None:
                mask_pil = Image.open(masks[k]).convert('L')
                mask_pil = mask_pil.point(lambda x: 255 if x > 0 else 0, '1')
                mask_arr = np.asarray(mask_pil).astype(np.float32)
            else:
                mask_arr = np.zeros(image.shape[:2], dtype=np.float32)
            total_masks.append(mask_arr)
        
        mask = np.stack(total_masks, axis = -1)

        if self.is_gray:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype('uint8')

        if self.transform is not None:
            results = self.transform(image=image, mask=mask[..., 0], mask1=mask[..., 1], mask2=mask[..., 2], mask3=mask[..., 3])
            image, mask, mask1, mask2, mask3 = results['image'], results['mask'], results['mask1'], results['mask2'], results['mask3']
            mask = np.stack([mask, mask1, mask2, mask3], axis=-1)

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
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image).astype('uint8')
        result['image'] = image
        if self.masks is not None:  
            mask = Image.open(self.masks[index]).convert('L')
            mask = mask.point(lambda x: 255 if x > 0 else 0, '1')
            mask = np.asarray(mask).astype(np.uint8)
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
    img_dir = '../data/raw/IDRiD/1. Original Images/a. Training Set'
    mask_dir = '../data/raw/IDRiD/2. All Segmentation Groundtruths/a. Training Set'
    imgs = list(Path(img_dir).glob('*.jpg'))
    # print(imgs)
    import albumentations as A
    t = A.Compose([
            A.Resize(1024, 1024),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.7),
            A.IAAAdditiveGaussianNoise()

    ], additional_targets={
        'mask1': 'mask',
        'mask2': 'mask',
        'mask3': 'mask'
    })

    ds = MultiLesionSegmentation(imgs, False, mask_dir, transform=t)

    print(len(ds))
    dl = DataLoader(ds, batch_size=4)
    batch = next(iter(dl))
    mask = batch['mask']

    for m in mask:
        print(m.shape)
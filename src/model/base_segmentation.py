import argparse
from catalyst.contrib.callbacks import DrawMasksCallback
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.dl import SupervisedRunner
import torch.optim as optim
import torch.nn as nn
from catalyst.contrib.nn import DiceLoss, IoULoss
import segmentation_models_pytorch as smp
import collections
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import re
import albumentations as A
from typing import List
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os
import torch
import catalyst
from skimage.io import imread as mask_read
from catalyst.contrib.utils.cv import image as cata_image
from catalyst import utils
from albumentations.pytorch import ToTensor


class LesionSegmentation(Dataset):
    def __init__(self, images: List[Path], masks: List[Path], transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]

        image = cata_image.imread(image_path)
        results = {'image': image}

        if self.masks is not None:
            mask = mask_read(self.masks[index]).astype(np.float32)
            results['mask'] = mask

        if self.transform is not None:
            results = self.transform(**results)

        results['filaname'] = image_path.name

        return results


class Transform(object):

    def __init__(self, image_size: int = 1024, preprocessing_fn=None):
        self.image_size = image_size
        self.preprocessing_fn = preprocessing_fn

    def pre_transform(self):
        return [
            A.Resize(self.image_size,
                     self.image_size, always_apply=True)
        ]

    def hard_transform(self):
        result = [
            A.RandomRotate90(),
            A.Cutout(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            A.GridDistortion(p=0.3),
            A.HueSaturationValue(p=0.3)
        ]

        return result

    def resize_transforms(self):
        pre_size = int(self.image_size * 2)

        random_crop = A.Compose([
            A.SmallestMaxSize(pre_size, p=1),
            A.RandomCrop(
                self.image_size, self.image_size, p=1
            )
        ])

        rescale = A.Compose(
            [A.Resize(self.image_size, self.image_size, p=1)])

        random_crop_big = A.Compose([
            A.LongestMaxSize(pre_size, p=1),
            A.RandomCrop(
                self.image_size, self.image_size, p=1
            )
        ])

        # Converts the image to a square of size self.image_size x self.image_size
        result = [
            A.OneOf([
                random_crop,
                rescale,
                random_crop_big
            ], p=1)
        ]

        return result

    def post_transform(self):
        return [A.Lambda(image=self.preprocessing_fn), ToTensor()]

    def _get_compose(self, transform):
        result = A.Compose([
            item for sublist in transform for item in sublist
        ])

        return result

    def train_transform(self):
        return self._get_compose([
            self.resize_transforms(),
            self.hard_transform(),
            self.post_transform()
        ])

    def validation_transform(self):
        return self._get_compose([
            self.pre_transform(),
            self.post_transform()
        ])

    def test_transform(self):
        return self.validation_transform()


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


def get_loader(
    images: List[Path],
    masks: List[Path],
    random_state: int,
    valid_size: float = 0.2,
    batch_size: int = 4,
    num_workers: int = 4,
    train_transforms_fn=None,
    valid_transforms_fn=None,
):
    indices = np.arange(len(images))

    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True)

    np_images = np.array(images)
    np_masks = np.array(masks)

    train_dataset = LesionSegmentation(
        np_images[train_indices].tolist(),
        np_masks[train_indices].tolist(),
        transform=train_transforms_fn
    )

    valid_dataset = LesionSegmentation(
        np_images[valid_indices].tolist(),
        np_masks[valid_indices].tolist(),
        transform=valid_transforms_fn
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    loaders = collections.OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    return loaders


def main(args):
    DEVICE = utils.get_device()
    if torch.cuda.is_available():
        print('GPU is available')
        print(f'Number of available gpu: {torch.cuda.device_count()} GPU')
        print(f'GPU: {DEVICE}')
    else:
        print('Oops! sorry dude, it`s seem like you have to use CPU instead :))')

    is_fp16_used = args['fp16']

    if is_fp16_used:
        batch_size = 4
    else:
        batch_size = 2

    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'

    # Model return logit values
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=None,
        in_channels=3,
        classes=1
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    img_paths, mask_paths = get_datapath(
        train_image_dir, train_mask_dir, lesion_type=args['type'])

    transforms = Transform(1024, preprocessing_fn)
    train_transform = transforms.train_transform()
    valid_transform = transforms.validation_transform()

    loaders = get_loader(
        images=img_paths,
        masks=mask_paths,
        random_state=SEED,
        batch_size=batch_size,
        train_transforms_fn=train_transform,
        valid_transforms_fn=valid_transform
    )

    criterion = {
        'dice': DiceLoss(),
        'iou': IoULoss(),
        'bce': nn.BCEWithLogitsLoss()
    }

    learning_rate = args['lr']
    num_epochs = args['epochs']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.25, patience=2)

    logdir = os.path.join(OUTDIR, args['type']+'_lr' + str(args['lr']) + '_ep' + str(args['epochs']))

    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)

    if is_fp16_used:
        fp16_params = dict(opt_level="O1")  # params for FP16
    else:
        fp16_params = None

    print(f"FP16 params: {fp16_params}")

    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    runner = SupervisedRunner(
        device=DEVICE, input_key="image", input_target_key="mask")

    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),

        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
        ),

        # metrics
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask"),
        # visualization
        DrawMasksCallback(output_key='logits',
                          input_image_key='image',
                          input_mask_key='mask',
                          summary_step=50
                          )
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs
        logdir=logdir,
        num_epochs=num_epochs,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=fp16_params,
        # prints train logs
        verbose=True,
    )


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--type', required=True, type=str)
    parse.add_argument('--fp16', default=True, type=bool)
    parse.add_argument('--lr', default=0.001, type=float)
    parse.add_argument('--epochs', default=20, type=int)

    args = vars(parse.parse_args())

    print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}, 'segmenttation pytorch version: {smp.__version__}")

    SEED = 42
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=True)

    ROOT = Path('../../data/raw')

    train_image_dir = ROOT / '1. Original Images' / 'a. Training Set'
    train_mask_dir = ROOT / '2. All Segmentation Groundtruths' / 'a. Training Set'
    test_image_dir = ROOT / '1. Original Images' / 'b. Testing Set'
    test_mask_dir = ROOT / '2. All Segmentation Groundtruths' / 'b. Testing Set'

    lesion_paths = {
        'MA': '1. Microaneurysms',
        'EX': '3. Hard Exudates',
        'HE': '2. Haemorrhages',
        'SE': '4. Soft Exudates'
    }

    OUTDIR = '../../models/base_segmentation'

    main(args)

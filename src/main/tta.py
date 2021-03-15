import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import ttach as tta
from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import os
from pathlib import Path
import numpy as np

import sys
sys.path.append('..')

from base_utils import save_output as so
from data import TestSegmentation
from data import EasyTransform
from config import TestConfig

import argparse


def get_model(params, model_name):
    # Model return logit values
    model = getattr(smp, model_name)(
        **params
    )

    return model

def test_tta(config, logdir):
    test_img_dir = Path(config['test_img_paths'])
    TEST_IMAGES = sorted(test_img_dir.glob("*.jpg"))

    # Model return logit values
    model = get_model(model_name=config['model_name'], params=config['model'])

    transform = EasyTransform(1024)
    augmentation = transform.resize_transforms() + [A.Normalize(), ToTensorV2()]
    test_transform = A.Compose(augmentation)

    # create test dataset
    test_dataset = TestSegmentation(
        TEST_IMAGES,
        transform=test_transform,
    )

    num_workers: int = 4

    infer_loader = DataLoader(
        test_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    checkpoints = torch.load(f"{logdir}/checkpoints/best.pth")
    model.load_state_dict(checkpoints['model_state_dict'])

    # D4 makes horizontal and vertical flips + rotations for [0, 90, 180, 270] angels.
    # and then merges the result masks with merge_mode="mean"
    tta_model = tta.SegmentationTTAWrapper(
        model, tta.aliases.d4_transform(), merge_mode="mean")

    tta_runner = SupervisedRunner(
        model=tta_model,
        device=utils.get_device(),
        input_key="image"
    )

    # this get predictions for the whole loader
    tta_predictions = []
    for batch in infer_loader:
        tta_pred = tta_runner.predict_batch(batch)
        tta_predictions.append(tta_pred['logits'].cpu().numpy())

    tta_predictions = np.vstack(tta_predictions)

    threshold = 0.5 #Need to choose best threshold

    for i, (features, logits) in enumerate(zip(test_dataset, tta_predictions)):
        image_name = features['filename']
        mask_ = torch.from_numpy(logits[0]).sigmoid()
        mask = utils.detach(mask_ > threshold).astype(np.float32)

        out_path = Path(config['out_dir']) / 'tta' / config['lesion_type'] / Path(logdir).name
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        out_name = out_path / image_name
        so(mask, out_name)  # PIL Image format

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--logdir', required=True, help='Path to where the model checkpoint is saved')
    args = vars(parse.parse_args())

    config = TestConfig.get_all_attributes()

    test_tta(config, args['logdir'])

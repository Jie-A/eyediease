import argparse
from albumentations.augmentations.transforms import Lambda
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
from PIL import Image
import cv2
from tqdm.auto import tqdm

import rasterio
from rasterio.windows import Window

import sys
sys.path.append('..')

from config import TestConfig
from data import EasyTransform
from data import TestSegmentation
from util import make_grid, save_output as so
import models

def get_model(params, model_name):
    # Model return logit values
    model = getattr(smp, model_name)(
        **params
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(params['encoder_name'], params['encoder_weights'])
    
    return model, preprocessing_fn

def str_2_bool(value: str):
    return True if value == 'true' else False

def test_tta(config, args):
    test_img_dir = Path(config['test_img_paths'])
    TEST_IMAGES = sorted(test_img_dir.glob("*.jpg"))

    # Model return logit values
    model = get_model(
        model_name=config['model_name'], params=config['model_params'])

    transform = EasyTransform(1024)
    augmentation = transform.resize_transforms() + \
        [A.Normalize(), ToTensorV2()]
    test_transform = A.Compose(augmentation)

    # create test dataset
    test_dataset = TestSegmentation(
        TEST_IMAGES,
        transform=test_transform,
    )

    num_workers: int = 2

    infer_loader = DataLoader(
        test_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    checkpoints = torch.load(f"{args['logdir']}/checkpoints/best.pth")
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

    if args['createprob']:
        for i, (features, logits) in tqdm(enumerate(zip(test_dataset, tta_predictions))):
            image_name = features['filename']
            mask_ = torch.from_numpy(logits[0]).sigmoid()
            mask_arr = mask_.numpy()

            out_path = Path(config['out_dir']) / 'tta' / \
                config['lesion_type'] / 'prob_image' / \
                Path(args['logdir']).name
            if not os.path.isdir(out_path):
                os.makedirs(out_path, exist_ok=True)

            out_name = out_path / image_name

            predicted_save = Image.fromarray(
                (mask_arr*255).astype('uint8'))
            predicted_save.save(out_name, "JPEG", quality=100)

    else:
        threshold = args['optim_thres']  # Need to choose best threshold
        for i, (features, logits) in enumerate(zip(test_dataset, tta_predictions)):
            image_name = features['filename']
            mask_ = torch.from_numpy(logits[0]).sigmoid()
            mask = utils.detach(mask_ > threshold).astype(np.float32)

            out_path = Path(config['out_dir']) / 'tta' / \
                config['lesion_type'] / Path(args['logdir']).name
            if not os.path.isdir(out_path):
                os.makedirs(out_path, exist_ok=True)

            out_name = out_path / image_name
            so(mask, out_name)  # PIL Image format


def tta_patches(config, args):
    test_img_dir = Path(config['test_img_paths'])
    TEST_IMAGES = sorted(test_img_dir.glob("*.jpg"))
    # TEST_IMAGES = [path for path in TEST_IMAGES if int(path.name.split('_')[1].split('.')[0]) > 67]

    if hasattr(smp, config['model_name']):
        model, preprocessing_fn = get_model(
            config['model_params'], config['model_name'])
    elif config['model_name'] == "TransUnet":
        from self_attention_cv.transunet import TransUnet

        model = TransUnet(**config['model_params'])
        preprocessing_fn = models.get_preprocessing_fn(pretrained=None)
    else:
        model, preprocessing_fn = models.get_model(
            model_name=config['model_name'], 
            params = config['model_params'])

    checkpoints = torch.load(f"{args['logdir']}/checkpoints/best.pth")
    model.load_state_dict(checkpoints['model_state_dict'])
    model = model.to(utils.get_device())
    model.eval()

    tta_model = tta.SegmentationTTAWrapper(
        model, tta.aliases.d4_transform(), merge_mode="mean")

    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Lambda(image = preprocessing_fn),
        ToTensorV2()
    ])

    for img in tqdm(TEST_IMAGES):
        with rasterio.open(img.as_posix(), transform=rasterio.Affine(1, 0, 0, 0, 1, 0)) as dataset:
            slices = make_grid(dataset.shape, window=512, min_overlap=32)
            preds = np.zeros(dataset.shape, dtype=np.float32)

            for (x1, x2, y1, y2) in slices:
                image = dataset.read([1,2,3], window = Window.from_slices((x1, x2), (y1, y2)))
                image = np.moveaxis(image, 0, -1)
                image = test_transform(image=image)['image']
                image = image.float()
                
                with torch.no_grad():
                    image = image.to(utils.get_device())[None]

                    logit = tta_model(image)[0][0]
                    score_sigmoid = logit.sigmoid().cpu().numpy()
                    # print(np.unique(score_sigmoid, return_counts=True))
                    score_sigmoid = cv2.resize(score_sigmoid, (512, 512))

                    if not str_2_bool(args['createprob']):
                        preds[x1:x2, y1:y2] = (score_sigmoid > args['optim_thres'])
                    else:
                        preds[x1:x2, y1:y2] = (score_sigmoid)

            if str_2_bool(args['createprob']):
                out_path = Path(config['out_dir']) / 'tta' / \
                    config['lesion_type'] / 'prob_image' / \
                    Path(args['logdir']).name
                if not os.path.isdir(out_path):
                    os.makedirs(out_path, exist_ok=True)

                out_name = out_path / img.name

                predicted_save = Image.fromarray(
                    (preds*255).astype('uint8'))
                predicted_save.save(out_name, "JPEG", quality=100)
                print(f'Saved {img.name} to {str(out_path)}')
            else:
                out_path = Path(config['out_dir']) / 'tta' / \
                    config['lesion_type'] / Path(args['logdir']).name
                if not os.path.isdir(out_path):
                    os.makedirs(out_path, exist_ok=True)

                out_name = out_path / img.name
                so(preds, out_name)  # PIL Image format


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--logdir', required=True,
                       help='Path to where the model checkpoint is saved')
    parse.add_argument('--createprob', type=str, default='false',
                       help='Just create a prob mask not binary')
    parse.add_argument('--optim_thres', type=float, default=0.0,
                       help='The optimal threshold optain from auc-pr curve')
    args = vars(parse.parse_args())

    config = TestConfig.get_all_attributes()

    if config['data_type'] == 'all':
        test_tta(config, args)
    else:
        tta_patches(config, args)

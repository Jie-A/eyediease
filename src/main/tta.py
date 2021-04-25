"""
@author: Duy Le <leanhduy497@gmail.com>
"""

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import ttach as tta
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import to_numpy, image_to_tensor, tensor_from_rgb_image

from catalyst.dl import utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

import rasterio
from rasterio.windows import Window

from torch.cuda import amp
import gc
import re

from .aucpr import get_auc, plot_aucpr_curve
from ..data import NormalTransform
from ..data import TestSegmentation
from .util import lesion_dict, get_datapath, make_grid, save_output as so
from . import archs

import logging
logging.basicConfig(level=logging.INFO)

def get_model(params, model_name):
    # Model return logit values
    params['encoder_weights'] = None
    model = getattr(smp, model_name)(
        **params
    )
    return model

def str_2_bool(value: str):
    if value.lower() in ['1', 'true']:
        return True 
    elif value.lower() in ['0', 'false']:
        return False
    else:
        raise ValueError(f'Invalid value, should be one of these 1, true, 0, false')

def test_tta(logdir, config, args):
    test_img_dir = config['test_img_path']
    test_mask_dir = config['test_mask_path'] 
    img_paths, mask_paths = get_datapath(test_img_dir, test_mask_dir, lesion_type=config['lesion_type'])

    # Model return logit values
    if hasattr(smp, config['model_name']):
        model = get_model(
            config['model_params'], config['model_name'])
    elif config['model_name'] == "TransUnet":
        from self_attention_cv.transunet import TransUnet
        model = TransUnet(**config['model_params'])
    else:
        model = archs.get_model(
            model_name=config['model_name'], 
            params = config['model_params'])
            
    preprocessing_fn, mean, std = archs.get_preprocessing_fn(dataset_name=config['dataset_name'])

    transform = NormalTransform(config['scale_size'])
    augmentation = transform.resize_transforms() + [A.Lambda(image=preprocessing_fn), ToTensorV2()]

    test_transform = A.Compose(augmentation)

    test_ds = TestSegmentation(img_paths, mask_paths, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=True)

    checkpoints = torch.load(f"{logdir}/checkpoints/{'best' if str_2_bool(args['best']) else 'last'}.pth")
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()
    model = model.to(utils.get_device())

    # D4 makes horizontal and vertical flips + rotations for [0, 90, 180, 270] angels.
    # and then merges the result masks with merge_mode="mean"
    tta_transform = getattr(tta.aliases, args['tta'] + '_transform')
    if args['tta'] == 'multiscale':
        param = {'scales': [1,2,4]}
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(**param), merge_mode="mean")
    else:
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(), merge_mode="mean")
    
    tta_predictions = []
    gt_masks = []
    filenames = []
    # this get predictions for the whole loader
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch['image'].to('cuda'))
            pred = pred.detach().cpu()
            pred = torch.sigmoid(pred)[0]
            pred = pred.squeeze(dim=0).numpy()
            tta_predictions.append(pred)
            gt_masks.append(batch['mask'][0].numpy())
            filenames.append(batch['filename'][0])
    
    # for i, (image_path, probs) in tqdm(enumerate(zip(TEST_IMAGES, tta_predictions))):
    #     image_name = image_path.name
    #     mask_arr = probs.astype('float32')

    #     assert str(mask_arr.dtype) == 'float32'
    #     out_path = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
    #         config['lesion_type'] / 'prob_image' / Path(logdir).name
        
    #     if not os.path.isdir(out_path):
    #         os.makedirs(out_path, exist_ok=True)

    #     out_name = out_path / image_name

    #     predicted_save = Image.fromarray(
    #         (mask_arr*255).astype('uint8'))
    #     predicted_save.save(out_name, "JPEG", quality=100)
    #     print(f'Saved {image_name} to {str(out_path)}')
    logging.info('====> Estimate auc-pr score')
    mean_auc = get_auc(gt_masks, tta_predictions, config)
    logging.info(f'MEAN-AUC {mean_auc}')
    logging.info('====> Find optimal threshold from 0 to 1 w.r.t auc-pr curve')
    optim_thres1, optim_thres2 = plot_aucpr_curve(
                                                gt_masks, 
                                                tta_predictions, 
                                                Path(logdir).name, 
                                                config)
    
    logging.info(f"Optimal threshold is {optim_thres1}")
    logging.info('====> Output binary mask base on optimal threshold value')
    # prob_dir =  Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
    #         config['lesion_type'] / 'prob_image' / Path(logdir).name
    # threshold = args['optim_thres']  # Need to choose best threshold
    for mask_name, pred_mask in tqdm(zip(filenames, tta_predictions)):
        mask = (pred_mask > optim_thres1).astype(np.uint8)
        out_path = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
            config['lesion_type'] / Path(logdir).name
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        out_name = out_path / mask_name
        so(mask, out_name)  # PIL Image format

    del gt_masks, tta_predictions, filenames
    gc.collect()
    logging.info('====> Finishing inference')

def tta_patches(logdir, config, args):
    test_img_dir = config['test_img_path']
    test_mask_dir = config['test_mask_path'] / lesion_dict[config['lesion_type']].dir_name
    TEST_MASKS =  sorted(test_mask_dir.glob("*.*"))

    if hasattr(smp, config['model_name']):
        model = get_model(
            config['model_params'], config['model_name'])
    elif config['model_name'] == "TransUnet":
        from self_attention_cv.transunet import TransUnet
        model = TransUnet(**config['model_params'])
    else:
        model = archs.get_model(
            model_name=config['model_name'], 
            params = config['model_params'])
    preprocessing_fn, _, _ = archs.get_preprocessing_fn(dataset_name=config['dataset_name'])
    
    checkpoints = torch.load(f"{logdir}/checkpoints/{'best' if str_2_bool(args['best']) else 'last'}.pth")
    model.load_state_dict(checkpoints['model_state_dict'])
    model = model.to(utils.get_device())
    model.eval()

    tta_transform = getattr(tta.aliases, args['tta'] + '_transform')
    if args['tta'] == 'multiscale':
        param = {'scales': [1,2,4]}
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(**param), merge_mode="mean")
    else:
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(), merge_mode="mean")
    
    resize_size = config['scale_size']
    test_transform = A.Compose([
        A.Resize(resize_size, resize_size),
        A.Lambda(image = preprocessing_fn),
        ToTensorV2()
    ])

    tta_predictions = []
    filenames = []
    for mask_path in tqdm(TEST_MASKS):
        img = test_img_dir / re.sub('_' + config['lesion_type'] + '.tif', '.jpg', mask_path.name)
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

                    logit = model(image)[0][0]
                    score_sigmoid = logit.sigmoid().cpu().numpy()
                    score_sigmoid = cv2.resize(score_sigmoid, (resize_size*2, resize_size*2), interpolation=cv2.INTER_LINEAR)

                    preds[x1:x2, y1:y2] = score_sigmoid
            tta_predictions.append(preds)
            filenames.append(str(mask_path.name))
            # out_path = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
            #     config['lesion_type'] / 'prob_image' / Path(logdir).name
            
            # if not os.path.isdir(out_path):
            #     os.makedirs(out_path, exist_ok=True)

            # out_name = out_path / img.name

            # predicted_save = Image.fromarray(
            #     (preds*255).astype('uint8'))
            # predicted_save.save(out_name, "JPEG", quality=100)
            # print(f'Saved {img.name} to {str(out_path)}')
    logging.info('====> Estimate auc-pr score')
    mean_auc = get_auc(None, tta_predictions, config)
    logging.info(f'MEAN-AUC {mean_auc}')
    logging.info('====> Find optimal threshold from 0 to 1 w.r.t auc-pr curve')
    optim_thres1, optim_thres2 = plot_aucpr_curve(None, tta_predictions, Path(logdir).name, config)
    
    # prob_dir = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
    #                 config['lesion_type'] / 'prob_image' / Path(logdir).name
    for mask_name, mask_pred in tqdm(zip(filenames, tta_predictions)):
        mask = (mask_pred > optim_thres1).astype(np.float32)

        out_path = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
            config['lesion_type'] / Path(logdir).name

        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        out_name = out_path / mask_name
        so(mask, out_name)  # PIL Image format

    del filenames, tta_predictions
    gc.collect()
    logging.info('====> Finishing inference')


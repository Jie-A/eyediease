import sys
sys.path.append('..')

import base_utils
from data import OneLesionSegmentation, MultiLesionSegmentation, CLASS_COLORS, CLASS_NAMES
from data import MediumTransform as Transform
from config import BaseConfig
from losses import get_loss
from optim import get_optimizer
from scheduler import get_scheduler

import os
from typing import List
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from functools import partial

from iglovikov_helper_functions.dl.pytorch import add_weight_decay
from catalyst import utils
from catalyst.dl import SupervisedRunner, CriterionCallback, EarlyStoppingCallback, SchedulerCallback, MetricAggregationCallback, IouCallback, DiceCallback
from catalyst.dl.callbacks import WandbLogger
from catalyst.contrib.nn import OneCycleLRWithWarmup

from pytorch_toolbelt.utils.catalyst import (
    HyperParametersCallback,
    draw_binary_segmentation_predictions,
    draw_multilabel_segmentation_predictions,
    ShowPolarBatchesCallback,
)
from pytorch_toolbelt.utils import count_parameters
from pytorch_toolbelt.utils.random import set_manual_seed

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau

import segmentation_models_pytorch as smp

import numpy as np
from sklearn.model_selection import train_test_split


def get_model(params, model_name):
    # Model return logit values
    model = getattr(smp, model_name)(
        **params
    )

    if params['encoder_weights'] is None:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            params['encoder_name'], "imagenet")
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            params['encoder_name'], params['encoder_weights'])

    return model, preprocessing_fn


def get_loader(
    images: List[Path],
    mask_dir: str,
    random_state: int,
    valid_size: float = 0.2,
    batch_size: int = 4,
    val_batch_size: int = 8,
    num_workers: int = 4,
    train_transforms_fn=None,
    valid_transforms_fn=None,
    preprocessing_fn=None,
    masks: List[Path]=None,
    mode='binary'
):
    indices = np.arange(len(images))

    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True)

    np_images = np.array(images)

    if mode == 'binary':
        np_masks = np.array(masks)
        
        train_dataset = OneLesionSegmentation(
            np_images[train_indices].tolist(),
            masks = np_masks[train_indices].tolist(),
            transform=train_transforms_fn,
            preprocessing_fn=preprocessing_fn
        )

        valid_dataset = OneLesionSegmentation(
            np_images[valid_indices].tolist(),
            masks = np_masks[valid_indices].tolist(),
            transform=valid_transforms_fn,
            preprocessing_fn=preprocessing_fn
        )
    else:
        train_dataset = MultiLesionSegmentation(
            np_images[train_indices].tolist(),
            mask_dir=mask_dir,
            transform=train_transforms_fn,
            preprocessing_fn=preprocessing_fn
        )

        valid_dataset = MultiLesionSegmentation(
            np_images[valid_indices].tolist(),
            mask_dir=mask_dir,
            transform=valid_transforms_fn,
            preprocessing_fn=preprocessing_fn
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    loaders = OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    log_info = [['train', 'valid'], [[len(train_loader), len(valid_loader)], [len(train_dataset), len(valid_dataset)]]]

    return loaders, log_info


def main(configs, seed):
    TRAIN_IMG_DIRS = Path(configs['train_img_path'])
    TRAIN_MASK_DIRS = Path(configs['train_mask_path'])

    # Get model
    model, preprocessing_fn = get_model(configs['model'], configs['model_name'])
    
    #Define transform (augemntation)
    transforms = Transform(
        configs['scale_size'],
        preprocessing_fn=preprocessing_fn
    )

    train_transform = transforms.train_transform()
    val_transform = transforms.validation_transform()
    preprocessing = transforms.get_preprocessing()

    if configs['data_mode'] == 'binary':
        ex_dirs, mask_dirs = base_utils.get_datapath(
            img_path=TRAIN_IMG_DIRS, mask_path=TRAIN_MASK_DIRS, lesion_type=configs['lesion_type'])

        base_utils.log_pretty_table(['full_img_paths', 'full_mask_paths'], [[len(ex_dirs), len(mask_dirs)]])
    elif configs['data_mode'] == 'multiclass':
        pass
    else:
        ex_dirs = list(TRAIN_IMG_DIRS.glob('*.jpg'))
        mask_dirs = None
    
    # Get data loader
    loader, log_info = get_loader(
        images=ex_dirs, 
        masks=mask_dirs, 
        mask_dir=TRAIN_MASK_DIRS,
        random_state=seed, 
        batch_size=configs['batch_size'], 
        val_batch_size=configs['val_batch_size'],
        num_workers= 4,
        train_transforms_fn=train_transform,
        valid_transforms_fn=val_transform,
        preprocessing_fn=preprocessing,
        mode=configs['data_mode']
    )

    base_utils.log_pretty_table(log_info[0], log_info[1])

    if configs['finetune']:
        #Free all weights in the encoder of model
        bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm
        for param in model.encoder.parameters():
            param.requires_grad = False

        #Disable batchnorm update 
        for m in model.encoder.modules():
            if isinstance(m, bn_types):
                m.track_running_stats = False
    
    print(f'[INFO] total and trainable parameters in the model {count_parameters(model)}')
    
    #Define training configurations
    #Dont set weight decay for batchnorm
    parameters = add_weight_decay.add_weight_decay(
        model, weight_decay=configs['weight_decay'])
    #Set optimizer
    optimizer = get_optimizer(configs['optimizer'], parameters, configs['learning_rate'], configs['weight_decay'])
    #Set learning scheduler
    scheduler = get_scheduler(
        configs['scheduler'], optimizer, configs['learning_rate'], configs['num_epochs'], \
        batches_in_epoch = len(loader['train']), mode= configs['mode']
    )
    #Set loss 
    criterion = {}
    for loss_name in configs['criterion']:
        loss_fn = get_loss(loss_name)
        criterion[loss_name] =loss_fn

    #Define callbacks
    callbacks = []
    losses = []
    for loss_name, loss_weight in configs['criterion'].items():
        criterion_callback = CriterionCallback(
            input_key="mask", 
            output_key="logits",
            criterion_key=loss_name, 
            prefix="loss_"+loss_name,
            multiplier= float(loss_weight)
        )

        callbacks.append(criterion_callback)
        losses.append(criterion_callback.prefix)
    
    callbacks += [MetricAggregationCallback(
        prefix="loss",
        mode="sum",
        metrics=losses
    )]

    if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
        callbacks += [SchedulerCallback(mode="batch")]
    elif isinstance(scheduler, (ReduceLROnPlateau)):
        callbacks += [SchedulerCallback(reduced_metric=configs['metric'])]

    hyper_callbacks = HyperParametersCallback(configs)

    if configs['data_mode'] == 'binary':
        visualize_predictions = partial(
            draw_binary_segmentation_predictions, image_key="image", targets_key="mask"
        )
    elif configs['data_mode'] == 'multilabel':
        visualize_predictions = partial(
            draw_multilabel_segmentation_predictions, image_key="image", targets_key="mask", class_colors = CLASS_COLORS
        )

    show_batches = ShowPolarBatchesCallback(
        visualize_predictions, metric=configs['metric'], minimize=True)

    early_stopping = EarlyStoppingCallback(
        patience=10, metric=configs['metric'], minimize=False)

    iou_scores = IouCallback(
        input_key="mask",
        activation="Sigmoid",
        per_class=False,
        class_args= CLASS_NAMES
    )

    dice_scores = DiceCallback(
        input_key="mask",
        activation="Sigmoid",
        per_class=False,
        class_args=CLASS_NAMES
    )
    #End define

    current_time = datetime.now().strftime("%b%d_%H_%M")
    prefix = f"{configs['lesion_type']}/{current_time}"
    log_dir = os.path.join("../../models/", prefix)
    os.makedirs(log_dir, exist_ok=False)
    logger = WandbLogger(project="HaemorrhageSegmentation",
                         name=current_time)


    callbacks += [hyper_callbacks, early_stopping, iou_scores, dice_scores, show_batches, logger]
    
    # model training
    runner = SupervisedRunner(
        device=utils.get_device(), input_key="image", input_target_key="mask")

    if configs['is_fp16']:
        fp16_params = dict(amp=True)  # params for FP16
    else:
        fp16_params = None

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        logdir=log_dir,
        loaders=loader,
        num_epochs=configs['num_epochs'],
        scheduler=scheduler,
        main_metric=configs['metric'],
        minimize_metric=False,
        fp16=fp16_params,
        verbose=True,
    )


if __name__ == "__main__":
    SEED = 1999
    set_manual_seed(SEED)
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=False, benchmark=True)

    configs = BaseConfig.get_all_attributes()
    main(configs, SEED)

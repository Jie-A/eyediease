"""
@author: Duy Le <leanhduy497@gmail.com>
"""
from sklearn.model_selection import train_test_split
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_toolbelt.utils.catalyst import (
    HyperParametersCallback,
    draw_binary_segmentation_predictions,
    ShowPolarBatchesCallback,
    RocAucMetricCallback
)
from catalyst.contrib.nn import OneCycleLRWithWarmup
from catalyst import dl
from catalyst.contrib.callbacks.wandb_logger import WandbLogger
from catalyst.dl import SupervisedRunner, CriterionCallback, EarlyStoppingCallback, SchedulerCallback, MetricAggregationCallback, IouCallback, DiceCallback
from catalyst import utils
from functools import partial
from collections import OrderedDict
from pathlib import Path
from typing import List
import os
import json
import logging
logging.basicConfig(level=logging.INFO, format='')

from . import util
from .util import lesion_dict, AucPRMetricCallback
from ..data import *
from .scheduler import get_scheduler
from .optim import get_optimizer
from .losses import get_loss, WeightedBCEWithLogits
from ..data import OneLesionSegmentation, MultiLesionSegmentation, CLASS_COLORS, CLASS_NAMES
from . import archs

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
    ben_method = None,
    masks: List[Path] = None,
    mode='binary',
    data_type = 'tile'
):
    indices = np.arange(len(images))

    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True)

    np_images = np.array(images)

    if mode == 'binary':
        np_masks = np.array(masks)

        train_dataset = OneLesionSegmentation(
            np_images[train_indices].tolist(),
            masks=np_masks[train_indices].tolist(),
            transform=train_transforms_fn,
            preprocessing_fn=preprocessing_fn,
            ben_transform = ben_method,
            data_type = data_type
        )

        valid_dataset = OneLesionSegmentation(
            np_images[valid_indices].tolist(),
            masks=np_masks[valid_indices].tolist(),
            transform=valid_transforms_fn,
            preprocessing_fn=preprocessing_fn,
            ben_transform = ben_method,
            data_type = data_type
        )
    else:
        train_dataset = MultiLesionSegmentation(
            np_images[train_indices].tolist(),
            mask_dir=mask_dir,
            transform=train_transforms_fn,
            ben_transform = ben_method,
            preprocessing_fn=preprocessing_fn
        )

        valid_dataset = MultiLesionSegmentation(
            np_images[valid_indices].tolist(),
            mask_dir=mask_dir,
            transform=valid_transforms_fn,
            ben_transform = ben_method,
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

    log_info = [['train', 'valid'], [[len(train_loader), len(valid_loader)], [
        len(train_dataset), len(valid_dataset)]]]

    return loaders, log_info


def train_model(exp_name, configs, seed):
    torch.autograd.set_detect_anomaly(True)

    TRAIN_IMG_DIRS = configs['train_img_path']
    TRAIN_MASK_DIRS = configs['train_mask_path']

    # Get model
    use_smp = True
    if hasattr(smp, configs['model_name']):
        model, preprocessing_fn = get_model(
            configs['model_params'], configs['model_name'])
    elif configs['model_name'] == "TransUnet":
        from self_attention_cv.transunet import TransUnet
        model = TransUnet(**configs['model_params'])
        preprocessing_fn = archs.get_preprocessing_fn(pretrained=None)
        use_smp=False
    else:
        model, preprocessing_fn = archs.get_model(
            model_name=configs['model_name'], 
            params = configs['model_params'])
        use_smp = False

    #Define transform (augemntation)
    Transform = get_transform(configs['augmentation'])
    transforms = Transform(
        configs['scale_size'],
        preprocessing_fn=preprocessing_fn
    )

    train_transform = transforms.train_transform()
    val_transform = transforms.validation_transform()
    preprocessing = transforms.get_preprocessing()

    if configs['data_mode'] == 'binary':
        ex_dirs, mask_dirs = util.get_datapath(
            img_path=TRAIN_IMG_DIRS, mask_path=TRAIN_MASK_DIRS, lesion_type=configs['lesion_type'])

        util.log_pretty_table(['full_img_paths', 'full_mask_paths'], [
                                    [len(ex_dirs), len(mask_dirs)]])
    elif configs['data_mode'] == 'multiclass':
        pass
    else:
        ex_dirs = list(TRAIN_IMG_DIRS.glob('*.jpg'))
        mask_dirs = None

    # Get data loader
    if configs['use_ben_transform']:
        ben_transform = load_ben_color
    else:
        ben_transform = None
    loader, log_info = get_loader(
        images=ex_dirs,
        masks=mask_dirs,
        mask_dir=TRAIN_MASK_DIRS,
        random_state=seed,
        batch_size=configs['batch_size'],
        val_batch_size=configs['val_batch_size'],
        num_workers=2,
        train_transforms_fn=train_transform,
        valid_transforms_fn=val_transform,
        preprocessing_fn=preprocessing,
        ben_method = ben_transform,
        mode=configs['data_mode'],
        data_type=configs['data_type']
    )

    #Visualize on terminal
    util.log_pretty_table(log_info[0], log_info[1])

    if use_smp:
        if configs['finetune']:
            #Free all weights in the encoder of model
            for param in model.encoder.parameters():
                param.requires_grad = False
        if configs['encoder_weights'] is not None:
            bn_types = nn.BatchNorm2d
            #Disable batchnorm update
            for m in model.encoder.modules():
                if isinstance(m, bn_types):
                    m.eval()

    param_group = []
    if hasattr(model, 'encoder'):
        encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
        param_group += [{'params': encoder_params, 'lr': configs['learning_rate']}]        
    if hasattr(model, 'decoder'):
        decoder_params = filter(lambda p: p.requires_grad, model.decoder.parameters())
        param_group += [{'params': decoder_params}]        
    if hasattr(model, 'segmentation_head'):
        head_params = filter(lambda p: p.requires_grad, model.segmentation_head.parameters())
        param_group += [{'params': head_params}]        
    if len(param_group) == 0:
        param_group = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]

    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    count_parameters = {"total": total, "trainable": trainable}

    logging.info(
        f'[INFO] total and trainable parameters in the model {count_parameters}')
    
    #Set optimizer
    optimizer = get_optimizer(
        configs['optimizer'], param_group, configs['learning_rate_decode'], configs['weight_decay'])
    #Set learning scheduler
    scheduler = get_scheduler(
        configs['scheduler'], optimizer, configs['learning_rate'], configs['num_epochs'],
        batches_in_epoch=len(loader['train']), mode=configs['mode']
    )
    #Set loss
    criterion = {}
    for loss_name in configs['criterion']:
        if loss_name == 'wbce':
            pos_weights = torch.tensor(configs['pos_weights'], device=utils.get_device())
            loss_fn = WeightedBCEWithLogits(pos_weights=pos_weights)
        else:
            loss_fn = get_loss(loss_name)
        criterion[loss_name] = loss_fn

    #Define callbacks
    callbacks = []
    losses = []
    for loss_name, loss_weight in configs['criterion'].items():
        criterion_callback = CriterionCallback(
            input_key="mask",
            output_key="logits",
            criterion_key=loss_name,
            prefix="loss_"+loss_name,
            multiplier=float(loss_weight)
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
    # elif configs['data_mode'] == 'multilabel':
    #     visualize_predictions = partial(
    #         draw_multilabel_segmentation_predictions, image_key="image", targets_key="mask", class_colors=CLASS_COLORS
    #     )

    show_batches_1 = ShowPolarBatchesCallback(
        visualize_predictions, metric="iou", minimize=False)

    # show_batches_2 = ShowPolarBatchesCallback(
    #     visualize_predictions, metric="loss", minimize=True)

    early_stopping = EarlyStoppingCallback(
        patience=10, metric=configs['metric'], minimize=False)

    iou_scores = IouCallback(
        input_key="mask",
        activation="Sigmoid",
        threshold=0.5
    )

    dice_scores = DiceCallback(
        input_key="mask",
        activation="Sigmoid",
    threshold=0.5
    )

    aucpr_scores = AucPRMetricCallback(
        input_key="mask",
    )

    # aucroc_scores = RocAucMetricCallback(
    #     input_key="mask",
    # )
    #End define

    prefix = f"{configs['lesion_type']}/{exp_name}"
    log_dir = os.path.join("models/", configs['dataset_name'], prefix)
    os.makedirs(log_dir, exist_ok=False)
    logger = WandbLogger(project=lesion_dict[configs['lesion_type']].project_name,
                         name=exp_name)

    #Save config as JSON format
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        configs['train_img_path'] = str(configs['train_img_path'])
        configs['train_mask_path'] = str(configs['train_mask_path'])
        json.dump(configs, f)

    callbacks += [hyper_callbacks, early_stopping,
                  iou_scores, dice_scores, aucpr_scores, show_batches_1, logger]

    
    # class CustomRunner(dl.SupervisedRunner):
    #     def _handle_batch(self, batch):
    #         x, y = batch

    #         y_pred = self.model(x)
    #         pass

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
        timeit=True,
        fp16=fp16_params,
        resume=configs['resume_path'],
        verbose=True,
    )
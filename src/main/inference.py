import albumentations as A
from torch.utils.data import Dataset, DataLoader
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer, TileMerger
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
from sklearn.metrics import plot_precision_recall_curve, average_precision_score, auc, precision_recall_curve
import matplotlib.pyplot as plt
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm

import sys
sys.path.append('..')

from data import TestSegmentation
from config import TestConfig
from base_utils import get_datapath, log_pretty_table

def run_inference(mode, model, loader):
    if mode == "all":

        pass
    else:  # mode tile overlapping

        pass
    pass


def run_tta_inference(mode):
    if mode == "all":

        pass
    else:  # mode tile overlapping

        pass

    pass


def get_model(model_name, params):
    params['encoder_weights'] = None

    model = getattr(smp, model_name)(
        **params
    )
    return model


def metric_analysis():
    # PR AUC, Dice, IoU, Precision, Recall, F1

    pass


def choose_best_threshold(targets, scores):
    precision, recall, threshold = precision_recall_curve(targets, scores)

    auc_precision_recall = auc(recall, precision)

    plt.plot(recall, precision)
    plt.savefig('../../outputs/figures')

    return auc_precision_recall


if __name__ == "__main__":
    test_config = TestConfig.get_all_attributes()
    test_aug = A.Compose([
        A.Normalize(),
        ToTensorV2(always_apply=True, p=1.0)
    ])
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(test_config['model_name'], test_config['model'])

    logdir = '../../models/EX/Mar15_22_06/checkpoints/best.pth'

    best_checkpoints = torch.load(logdir)
    best_model = best_checkpoints['model_state_dict']
    model.load_state_dict(best_model)
    model = model.to(device)

    test_imgs, test_masks = get_datapath(test_config['train_img_path'], test_config['train_mask_path'], test_config['lesion_type'])
    
    print(log_pretty_table(['image', 'mask'], [[len(sorted(test_imgs)), len(sorted(test_masks))]]))

    test_loader = DataLoader(
        TestSegmentation(
            test_imgs,
            test_masks,
            transform=test_aug
        ),
        batch_size=1
    )

    with torch.no_grad():
        model.eval()
        # probs = []
        # masks = []
        sum_prauc = 0
        for i, result in enumerate(tqdm(test_loader)):
            img = result['image'].to(device)
            logit = model(img)
            prob = torch.sigmoid(logit)
            prob = prob.cpu().numpy()[0][0]
            mask = result['mask'].numpy()[0]
            # probs.append(prob)
            # masks.append(mask)
            prauc = average_precision_score(mask.reshape(-1), prob.reshape(-1))
            sum_prauc += prauc

        sum_prauc /= len(test_loader)

        # probs = np.array(probs).reshape(-1).astype(np.float32)
        # masks = np.array(masks).reshape(-1).astype(np.uint8)

        # prauc = choose_best_threshold(masks, probs)

        print('[INFO] auc-pr', sum_prauc)
        #Mar15_22_06 auc [INFO] auc-pr 0.3344325889668523

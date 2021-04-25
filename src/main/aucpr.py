from PIL import Image
import numpy as np
import os
import re
import sys
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from pathlib import Path
from tqdm .auto import tqdm
import plotly.express as px
import logging

logging.basicConfig(level=logging.INFO)

from .util import lesion_dict

def get_auc(gt_masks, pred_masks, config):    
    sum_pav = 0
    i =  0
    if gt_masks is None:
        gt_dir = config['test_mask_path'] / lesion_dict[config['lesion_type']].dir_name
        gt_masks = list(gt_dir.glob("*.*"))

    for gt_mask, pred_mask in tqdm(zip(gt_masks, pred_masks)):
        if not isinstance(gt_mask, np.ndarray):
            gt_mask = Image.open(gt_mask).convert('L')
            gt_mask = gt_mask.point(lambda x: 255 if x > 0 else 0, '1')
            gt_mask = np.asarray(gt_mask).astype(np.uint8)
        pav = average_precision_score(gt_mask.reshape(-1), pred_mask.reshape(-1))
        sum_pav += pav
        i += 1

    mpav = sum_pav / i
    return mpav

def plot_aucpr_curve(gts, preds, exp_name, test_config):
    # gt_dir = test_config['test_mask_path'] / lesion_dict[test_config['lesion_type']].dir_name
    # prob_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'] ,'tta', test_config['lesion_type'], 'prob_image', exp_name) 
    figure_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'], 'figures', test_config['lesion_type']) 

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    thresh_list = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    thresh_size = len(thresh_list)
    sn = np.empty(thresh_size, dtype=float)
    ppv = np.empty(thresh_size, dtype=float)
    thresh_array = np.array(thresh_list)

    if gts is None:
        gt_dir = test_config['test_mask_path'] / lesion_dict[test_config['lesion_type']].dir_name
        gts = list(gt_dir.glob("*.*"))

    for th in range(thresh_size):
        threshold = thresh_array[th]
        true_p=0
        actual_p=0
        pred_p=0    

        for gt_mask, pred_mask in tqdm(zip(gts, preds)):
            if not isinstance(gt_mask, np.ndarray):
                gt_mask = Image.open(gt_mask).convert('L')
                gt_mask = gt_mask.point(lambda x: 255 if x > 0 else 0, '1')
                gt_mask = np.array(gt_mask).astype('uint8')
            arr_pred = (pred_mask > threshold).astype('uint8')
            tp = np.sum(gt_mask & arr_pred)
            ap = np.sum(gt_mask)
            pp = np.sum(arr_pred)
            true_p += tp
            actual_p += ap
            pred_p += pp

        sn[th] = (float(true_p) + 1e-7)/(float(actual_p)+ 1e-7)
        ppv[th] = (float(true_p) +  1e-7)/(float(pred_p) + 1e-7)
    
    recall = np.array(sn)
    precision = np.array(ppv)
    aucpr = auc(recall, precision)
    #https://www.kaggle.com/nicholasgah/optimal-probability-thresholds-using-pr-curve
    optimal_threshold = sorted(list(zip(
        np.abs(precision - recall), thresh_list)), key=lambda i: i[0], reverse=False)[0][1]

    optimal_threshold_1 = sorted(list(zip(np.sqrt((1-precision)**2 + (1-recall)**2), thresh_list)), key=lambda i: i[0], reverse=False)[0][1]

    logging.info(f'OPTIMAL THRESHOLD: {optimal_threshold}')
    logging.info(f'OPTIMAL THRESHOLD 1: {optimal_threshold_1}')

    fig = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve AUC:{aucpr}',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.write_image(figure_dir + "/{}.jpg".format(str(exp_name)))
    logging.info(f'Saved AUC-PR Curve to {figure_dir}')

    return optimal_threshold, optimal_threshold_1

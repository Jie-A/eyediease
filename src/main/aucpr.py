from PIL import Image
import numpy as np
import os
import re
import sys
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path
from tqdm .auto import tqdm
import plotly.express as px
import logging

logging.basicConfig(level=logging.INFO)

# import sys
# sys.path.append('..')

from .util import lesion_dict

def get_auc(exp_name, test_config):    
    gt_dir = test_config['test_mask_paths'] / lesion_dict[test_config['lesion_type']].dir_name
    prob_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'] ,'tta', test_config['lesion_type'], 'prob_image', exp_name) 
    figure_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'], 'figures', test_config['lesion_type']) 

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    arr_gts = []
    arr_probs = []

    for image_path in tqdm(list(gt_dir.glob('*.tif'))):
        prob_name = re.sub('_' + test_config['lesion_type']+ '.tif', '.jpg', image_path.name)
        im_prob = Image.open(os.path.join(prob_dir,prob_name))
        im_prob = im_prob.resize((1024, 1024), resample=Image.BILINEAR)
        im_size = im_prob.size

        im_gt = Image.open(str(image_path))
        im_gt = im_gt.resize((1024, 1024), resample=Image.BILINEAR)
        
        arr_gt = np.asarray(im_gt).astype(np.uint8)

        if len(arr_gt.shape) == 3:
            continue

        assert len(arr_gt.shape) == 2

        arr_prob = (np.array(im_prob)).astype(float)/255

        assert len(arr_prob.shape) == 2

        arr_gts.append(arr_gt)
        arr_probs.append(arr_prob)

    arr_gts = np.array(arr_gts).reshape(-1)
    arr_probs = np.array(arr_probs).reshape(-1)

    precision, recall, thresholds = precision_recall_curve(arr_gts, arr_probs)
    auc_score = auc(recall, precision)

    #https://www.kaggle.com/nicholasgah/optimal-probability-thresholds-using-pr-curve
    optimal_threshold = sorted(list(zip(
        np.abs(precision - recall), thresholds)), key=lambda i: i[0], reverse=False)[0][1]

    optimal_threshold_1 = sorted(list(zip(np.sqrt((1-precision)**2 + (1-recall)**2), thresholds)), key=lambda i: i[0], reverse=False)[0][1]

    logging.info(f'OPTIMAL THRESHOLD: {optimal_threshold}')
    logging.info(f'OPTIMAL THRESHOLD 1: {optimal_threshold_1}')

    fig = px.area(
        x=recall, y=precision,
        title='Precision-Recall Curve (AUC={:.4f})'.format(auc_score),
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.write_image(figure_dir + "/{}.jpg".format(exp_name))
    logging.info(f'Saved AUC-PR Curve to {figure_dir}')

    return optimal_threshold, optimal_threshold_1

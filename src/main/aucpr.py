from PIL import Image
import numpy as np
import os
import re
import sys
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from pathlib import Path
from tqdm .auto import tqdm
import plotly.express as px

import sys
sys.path.append('..')

from util import lesion_dict
from config import TestConfig

test_config = TestConfig.get_all_attributes()

gt_dir = Path(test_config['test_mask_paths']) / lesion_dict[test_config['lesion_type']].dir_name
prob_dir = test_config['out_dir']  + '/tta/' + test_config['lesion_type'] + '/prob_image/Mar28_10_24/' 
figure_dir = test_config['out_dir'] + '/figures/' + test_config['lesion_type'] 

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

i = 0
sum_pav = 0

arr_gts = []
arr_probs = []

for image_path in tqdm(list(gt_dir.glob('*.tif'))):
    prob_name = re.sub('_' + test_config['lesion_type']+ '.tif', '.jpg', image_path.name)
    im_prob = Image.open(prob_dir+prob_name)
    im_size = im_prob.size

    im_gt = Image.open(str(image_path))
    im_gt = im_gt.resize(im_size)

    arr_gt = np.asarray(im_gt)
    arr_prob = (np.array(im_prob)).astype(float)/255

    arr_gts.append(arr_gt)
    arr_probs.append(arr_prob)

arr_gts = np.array(arr_gts).reshape(-1)
arr_probs = np.array(arr_probs).reshape(-1)

precision, recall, thresholds = precision_recall_curve(arr_gts, arr_probs)
auc_score = auc(recall, precision)

#https://www.kaggle.com/nicholasgah/optimal-probability-thresholds-using-pr-curve
optimal_proba_cutoff = sorted(list(zip(
    np.abs(precision - recall), thresholds)), key=lambda i: i[0], reverse=False)[0][1]

print(f'[INFO] OPTIMAL THRESHOLD: {optimal_proba_cutoff}')

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc_score:.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.write_image(figure_dir + '/Mar28_10_24.jpg')

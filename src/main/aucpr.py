from PIL import Image
import numpy as np
import os
import re
import sys
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from pathlib import Path
from tqdm .auto import tqdm
import plotly.express as px
import plotly

import sys
sys.path.append('..')

from base_utils import lesion_paths
from config import TestConfig

test_config = TestConfig.get_all_attributes()

gt_dir = Path(test_config['test_mask_paths']) / lesion_paths[test_config['lesion_type']]
prob_dir = test_config['out_dir']  + '/tta/' + test_config['lesion_type'] + '/prob_image/Mar26_13_36/' 
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

    prob_size = im_prob.size
    im_gt = Image.open(str(image_path))
    im_gt = im_gt.resize(prob_size)

    arr_gt = (np.array(im_gt)/255).astype(bool)
    arr_prob = (np.array(im_prob)).astype(float)/255

    arr_gts.append(arr_gt.reshape(-1))
    arr_probs.append(arr_prob.reshape(-1))

arr_gts = np.array(arr_gts).reshape(-1)
arr_probs = np.array(arr_probs).reshape(-1)

precision, recall, _ = precision_recall_curve(arr_gts, arr_probs)
auc_score = average_precision_score(arr_gts, arr_probs)

name = f"{test_config['lesion_type']} (AP={auc_score:.2f})"

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
fig.write_image(figure_dir + '/Mar26_13_36.jpg')

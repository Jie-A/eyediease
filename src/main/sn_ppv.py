from PIL import Image
import numpy as np
import os, sys
import re
import cv2
import albumentations as A

sys.path.append('..')
import argparse

from config import TestConfig
from base_utils import lesion_paths

parse = argparse.ArgumentParser()
parse.add_argument('--dir', required=True)

args = vars(parse.parse_args())

test_config = TestConfig.get_all_attributes()
transform = A.Compose([
	A.LongestMaxSize(
	test_config['scale_size']), 
	A.PadIfNeeded(test_config['scale_size'], test_config['scale_size'], border_mode=cv2.BORDER_CONSTANT, value=0)]
	)

test_size=27
gt_dir = test_config['test_mask_paths'] + \
	'/' + lesion_paths[test_config['lesion_type']]

pred_dir = test_config['out_dir'] + '/' + 'tta/' + args['dir']

sn = np.empty(test_size+1, dtype=float) 
ppv = np.empty(test_size+1, dtype=float)
sp = np.empty(test_size+1, dtype=float)
image_paths = np.empty(test_size+1, dtype=object)

i=0
for image_path in os.listdir(gt_dir):
	image_paths[i] = image_path
	im_gt = Image.open(gt_dir+'/'+image_path)
	arr_gt = np.array(im_gt)
	arr_gt = transform(image = arr_gt)['image']

	pred_image_path = re.sub('_' + test_config['lesion_type'] + '.tif', '.jpg', image_path)
	im_pred = Image.open(pred_dir+'/'+pred_image_path)
	im_binary = im_pred.convert('1')
	arr_pred = np.asarray(im_binary).astype(np.uint8)

	if len(arr_gt.shape) > 2:
		arr_gt = np.sum(arr_gt, axis=-1)

	true_p = np.sum(arr_gt & arr_pred)
	actual_p = np.sum(arr_gt)
	pred_p = np.sum(arr_pred)
	
	false_p = pred_p - true_p
	actual_n = test_config['scale_size']*test_config['scale_size'] - actual_p
	true_n = actual_n - false_p

	if actual_p == 0:
		sn[i] = 1
	else:
		sn[i] = float(true_p)/float(actual_p)
		
	if pred_p == 0:
		ppv[i] = 1
	else:
		ppv[i] = float(true_p)/float(pred_p)

	if actual_n == 0:
		sp[i] = 1
	else:
		sp[i] = float(true_n)/float(actual_n)
	i+=1

image_paths[i] = "Total"
sn[i] = np.mean(sn[:-1])
ppv[i] = np.mean(ppv[:-1])
sp[i] =np.mean(sp[:-1])

sn_csv = np.stack((image_paths,sn), axis=1)
ppv_csv = np.stack((image_paths,ppv), axis=1)
sp_csv = np.stack((image_paths,sp), axis=1)

save_dir = '../../outputs/result_assessment/' + args['dir']

if not os.path.exists(save_dir):
	os.makedirs(save_dir, exist_ok=True)

np.savetxt(f"{save_dir}/sn.csv", sn_csv, delimiter=",", fmt="%s")
np.savetxt(f"{save_dir}/ppv.csv", ppv_csv, delimiter=",", fmt="%s")
np.savetxt(f"{save_dir}/sp.csv", sp_csv, delimiter=",", fmt="%s")

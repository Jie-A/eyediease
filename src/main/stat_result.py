from PIL import Image
import numpy as np
import os, sys
import re
import cv2
import albumentations as A

sys.path.append('..')
import argparse

from config import TestConfig
from util import lesion_dict

parse = argparse.ArgumentParser()
parse.add_argument('--dir', required=True)
parse.add_argument('--eps', type=float, default=1e-7)

args = vars(parse.parse_args())

test_config = TestConfig.get_all_attributes()
transform = A.Compose([
	A.LongestMaxSize(
	test_config['scale_size']), 
	A.PadIfNeeded(test_config['scale_size'], test_config['scale_size'], border_mode=cv2.BORDER_CONSTANT, value=0)]
	)

gt_dir = test_config['test_mask_paths'] + \
	'/' + lesion_dict[test_config['lesion_type']].dir_name

test_size=len(os.listdir(gt_dir))
pred_dir = test_config['out_dir'] + '/' + 'tta/' + args['dir']

sn = np.empty(test_size+1, dtype=float) 
ppv = np.empty(test_size+1, dtype=float)
sp = np.empty(test_size+1, dtype=float)
dice = np.empty(test_size+1, dtype=float)
iou = np.empty(test_size+1, dtype=float)
image_paths = np.empty(test_size+1, dtype=object)

i=0
for image_path in os.listdir(gt_dir):
	image_paths[i] = image_path
	im_gt = Image.open(gt_dir+'/'+image_path)
	arr_gt = np.array(im_gt)
	if test_config['data_type'] == 'all':
		arr_gt = transform(image = arr_gt)['image']

	pred_image_path = re.sub('_' + test_config['lesion_type'] + '.tif', '.jpg', image_path)
	im_pred = Image.open(pred_dir+'/'+pred_image_path)
	if im_pred is None:
		continue
	im_binary = im_pred.convert('1')
	arr_pred = np.asarray(im_binary).astype(np.uint8)

	if len(arr_gt.shape) > 2:
		arr_gt = np.sum(arr_gt, axis=-1)

	true_p = np.sum(arr_gt & arr_pred)
	actual_p = np.sum(arr_gt)
	pred_p = np.sum(arr_pred)
	
	false_p = pred_p - true_p
	if test_config['data_type'] == 'all':
		actual_n = test_config['scale_size']*test_config['scale_size'] - actual_p
	else:
		actual_n = im_gt.size[0]*im_gt.size[1] - actual_p
	true_n = actual_n - false_p

	union = actual_p + false_p

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

	iou[i] = (true_p + args['eps']*(union == 0).astype('float')) / (actual_p + false_p + args['eps'])

	dice[i] = (2*true_p + args['eps']*(union == 0).astype('float')) / (true_p + actual_p + false_p + args['eps'])
	
	i+=1

image_paths[i] = "Avg:"
sn[i] = np.mean(sn[:-1])
ppv[i] = np.mean(ppv[:-1])
sp[i] =np.mean(sp[:-1])
iou[i] = np.mean(iou[:-1])
dice[i] = np.mean(dice[:-1])

sn_csv = np.stack((image_paths,sn), axis=1)
ppv_csv = np.stack((image_paths,ppv), axis=1)
sp_csv = np.stack((image_paths,sp), axis=1)
iou_csv = np.stack((image_paths, iou), axis=1)
dice_csv = np.stack((image_paths, dice), axis=1)

save_dir = '../../outputs/result_assessment/' + args['dir']

if not os.path.exists(save_dir):
	os.makedirs(save_dir, exist_ok=True)

np.savetxt(f"{save_dir}/sn.csv", sn_csv, delimiter=",", fmt="%s")
np.savetxt(f"{save_dir}/ppv.csv", ppv_csv, delimiter=",", fmt="%s")
np.savetxt(f"{save_dir}/sp.csv", sp_csv, delimiter=",", fmt="%s")
np.savetxt(f"{save_dir}/iou.csv", iou_csv, delimiter=",", fmt="%s")
np.savetxt(f"{save_dir}/dice.csv", dice_csv, delimiter=",", fmt="%s")

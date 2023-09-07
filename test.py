import numpy as np
import cv2
import os
import torch
import subprocess
import glob
from options.test_options import TestOptions
from model.net import SST_Inpainting_Model
from util.utils import generate_mask, getPthList
import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TestOptions().parse()
# print('config.dataset_path: ', config.dataset_path)
if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)

total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

print('configuring model..')
ourModel = SST_Inpainting_Model(in_ch=3, out_ch=3, opt=config)
# ourModel.print_networks()
if config.load_model_dir != '':
    print('Loading pretrained model from {}'.format(config.load_model_dir))
    print('os.path.join(config.load_model_dir, *.pth', (os.path.join(config.load_model_dir, '*.pth')))
    ourModel.load_networks(getPthList(os.path.join(config.load_model_dir, '*.pth')))
    print('Loading done.')

if config.random_mask:
    np.random.seed(config.seed)
map = cv2.imread('map.png', cv2.IMREAD_GRAYSCALE)
map[map == 255] = 1
map = np.float32(map)
for i in range(test_num):
    mask_map, mask_all = generate_mask(im_size=config.img_shapes, mask_map=map)
    image = cv2.imread(pathfile[i])
    cv2.imwrite(os.path.join(config.saving_path, '01truth_{:03d}.png'.format(i + 1)), image)

    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image_vis = image * (1 - mask_all) + 255 * mask_all
    image_vis = np.transpose(image_vis[0], [1, 2, 0])
    cv2.imwrite(os.path.join(config.saving_path, '02input_{:03d}.png'.format(i + 1)), image_vis.astype(np.uint8))
    result = ourModel.evaluate(image, mask_map, mask_all)
    mask_all[mask_all == 1] = 255
    cv2.imwrite(os.path.join(config.saving_path, '02mask_{:03d}.png'.format(i + 1)), mask_all[0][0])

    result = np.transpose(result[0], [1, 2, 0])
    cv2.imwrite(os.path.join(config.saving_path, '03output_{:03d}.png'.format(i + 1)), result)

    print(' > {} / {}'.format(i + 1, test_num))
print('done.')

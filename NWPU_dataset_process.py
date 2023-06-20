import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import cv2


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    if len(point) >= 4:
        kth = 3
    else:
        kth = len(point)-1
    square = np.sum(point * point, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0))
    if len(point) == 1:
        dis = np.zeros_like(point)
    else:
        dis = np.mean(np.partition(dis, kth, axis=1)[:, 1:kth + 1], axis=1, keepdims=True)
    return dis



root = '/media/user5/7C3637AF363768F2/Users/user5/Desktop/NWPU_dataset/mock_data'

# now generate the ShanghaiA's ground truth
train = os.path.join(root, 'train', 'images')
test = os.path.join(root, 'validation', 'images')

path_sets = [train, test]

min_size = 832
max_size = 1152

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    # for every image
    print('image path: ', img_path)

    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth'))
    img = cv2.imread(img_path)
    im_h, im_w, _ = img.shape

    gt = mat["annPoints"].astype(np.float32)
    if len(gt) >= 1:
        idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= im_w) * (gt[:, 1] >= 0) * (gt[:, 1] <= im_h)
        gt = gt[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    if rr != 1.0:
        img = cv2.resize(np.array(img), (im_w, im_h), cv2.INTER_CUBIC)
        gt = gt * rr
    if 'train' in img_path and len(gt) >= 1:
        dis = find_dis(gt)
        gt = np.concatenate((gt, dis), axis=1)

    im_save_path = img_path.replace('mock_data',
                                    'resized_mock_data')
    cv2.imwrite(im_save_path, img)
    gd_save_path = im_save_path.replace('jpg', 'npy').replace('images', 'ground_truth')
    np.save(gd_save_path, gt)
    #
    # while True:
    #     cv2.imshow('Demo', img)
    #     if cv2.waitKey(1) == 27:  # q to quit
    #         break

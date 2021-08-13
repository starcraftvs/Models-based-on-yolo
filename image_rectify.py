import math
import numpy as np
from pathlib import Path
import cv2
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

from constants import *
from utils.data.datasets import GetImgInf
from rectify_utils.rectify_utils import gen_homography, get_shift, find_dimensions


def get_img_label(folder_path):
    ext = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
    txt_ext = ('txt',)
    paths = []
    txt_paths = []

    for base_path, folder_list, file_list in os.walk(folder_path):
        for file_name in file_list:
            file_path = os.path.join(base_path, file_name)
            file_ext = file_path.rsplit('.', maxsplit=1)
            if len(file_ext) != 2:
                continue
            if file_ext[1] not in ext and file_ext[1] not in txt_ext:
                continue
            if file_ext[1] in ext:
                paths.append(file_path)
            if file_ext[1] in txt_ext:
                txt_paths.append(file_path)
    paths = sorted(paths)
    txt_paths = sorted(txt_paths)
    return paths, txt_paths


def correct_image(img, homography):
    if homography is None:
        return img
    (min_x, min_y, max_x, max_y) = find_dimensions(img, homography)
    # _, (min_x, min_y, max_x, max_y) = get_shift(img, homography)
    print("min_x, min_y, max_x, max_y", min_x, min_y, max_x, max_y)
    img_w = int(math.ceil(max_x - min_x))
    img_h = int(math.ceil(max_y - min_y))
    img_warp = cv2.warpPerspective(img, homography, (img_w, img_h))
    return img_warp


def rectify(im_path, param):
    param = np.array([float(item) for item in param])
    param = param * (UP_VALUE1-LOW_VALUE1) + LOW_VALUE1
    angle, shift_h, shift_v, shear = param
    angle=math.asin(angle)*180/math.pi
    shift_h=math.log(shift_h)
    shift_v=math.log(shift_v)
    im = Image.open(im_path)
    img = np.array(im)
    width, height = im.size
    hom = gen_homography(width, height, angle, shift_v, shift_h, shear)
    print('hom', hom)
    img_warped = correct_image(img, hom)
    # plt.imshow(img_warped)
    # plt.show()
    return img_warped,hom



def main():
    root_dir='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train'
    infs=GetImgInf(root_dir=root_dir,data_dir='img2test',label_dir='pred_label2')
    dst_folder=os.path.join(root_dir,'try1')
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for inf in infs:
        with open (inf['label_path'],'r') as f:
            param=list(map(float,f.readline().strip('[').strip(']').split(' ')))
            print(param)
            image_warped,hom = rectify(inf['img_path'], param)
            image_warped = cv2.cvtColor(image_warped, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(dst_folder,inf['img_path'].split('/')[-1]), image_warped)


if __name__ == "__main__":
    img_path = '/fast_data2/jtdata/train_ds/stich/210209/4test2/SWIRE_SNAPSHOT_IMAGE601cb6a3685361.46009138.jpg'
    pred_path = '/fast_data2/jtdata/train_ds/stich/210209/4test2/SWIRE_SNAPSHOT_IMAGE601cb6a3685361.46009138_pred.txt'
    pred_path = '/fast_data2/jtdata/train_ds/stich/210209/pred/SWIRE_SNAPSHOT_IMAGE601cb6a3685361.46009138.txt'
    # rectify(img_path, pred_path)
    # main('/fast_data2/jtdata/train_ds/stich/210209/4test2/')
    # main('/fast_data2/jtdata/train_ds/stich/210209/4test2-bg/')
    # main('/fast_data2/jtdata/train_ds/stich/210209/lm_test/costa')
    # main('/fast_data2/jtdata/tmp/rec')
    # main('/fast_data2/jtdata/train_ds/stich/210209/uniliver/')
    
    # main('/fast_data2/jtdata/train_ds/stich/210209/lm_test/sym')
    main()
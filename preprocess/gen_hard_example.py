#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/12/1
@Description:
"""

import os
import cv2
import numpy as np
import sys
sys.path.append('../')
npr = np.random

from tqdm import tqdm
from detector import Detector
from utils import cal_iou, convert_to_square, create_h5_box, create_h5_img

WIDER_FACE_IMG_DIR = '../data/WIDER_train/images'
WIDER_FACE_ANNO_FILE = '../data/wider_face_train_bbx_gt.txt'
DETECT_WIDEFACE_NUM = 3000


def load_widerface_dataset(images_dir, anno_file):
    data = dict()
    images_jpg = []
    bboxes = []

    num = 0
    with open(anno_file, 'r', encoding='utf-8')as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            image_fall_path = images_dir + '/' + line

            im = cv2.imread(image_fall_path, cv2.COLOR_BGR2RGB)
            images_jpg.append(im)
            face_num = int(f.readline().strip())
            faces = []
            for i in range(face_num):
                bb_info = f.readline().strip('\n').split(' ')
                x1, y1, w, h = [float(bb_info[i]) for i in range(4)]
                faces.append([x1, y1, x1 + w, y1 + h])
            bboxes.append(faces)

            num += 1
            if num >= DETECT_WIDEFACE_NUM:
                break

        data['images'] = images_jpg  # all image pathes
        data['boxes'] = bboxes  # all image bboxes

    return data


def save_hard_example(save_size, save_dir, data_gt, det_boxes):
    """将网络识别的box用来裁剪原图像作为下一个网络的输入"""
    img_list = data_gt['images']
    gt_boxes_list = data_gt['boxes']
    num_of_images = len(img_list)

    assert len(det_boxes) == num_of_images, "弄错了"

    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    positive_list = []
    negative_list = []
    part_list = []

    for img, dets, gts in tqdm(zip(img_list, det_boxes, gt_boxes_list)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if dets is None or dets.shape[0] == 0:
            continue
        # img = cv2.imread(im_idx)
        # 转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            iou = cal_iou(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size),
                                    interpolation=cv2.INTER_LINEAR)

            # 划分种类
            if np.max(iou) < 0.3 and neg_num < 60:
                negative_list.append(resized_im)
                n_idx += 1
                neg_num += 1
            else:
               
                idx = np.argmax(iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # 偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                roi = np.array([float(offset_x1), float(offset_y1), float(offset_x2), float(offset_y2)])

                # pos和part
                if np.max(iou) >= 0.65:
                    positive_list.append([resized_im, roi])
                    p_idx += 1

                elif np.max(iou) >= 0.4:
                    part_list.append([resized_im, roi])
                    d_idx += 1

    print('%s 个图片已处理，pos：%s  part: %s neg:%s' % (image_done, p_idx, d_idx, n_idx))

    base_num = 100000
    if len(negative_list) > base_num * 3:
        neg_keep = npr.choice(len(negative_list), size=base_num * 3, replace=True)
        negative_list = np.asarray(negative_list)[neg_keep]

    sum_p = len(negative_list) // 3
    pos_keep = npr.choice(len(positive_list), sum_p, replace=True)
    part_keep = npr.choice(len(part_list), sum_p, replace=True)

    positive_list = np.asarray(positive_list)[pos_keep]
    part_list = np.asarray(part_list)[part_keep]
    print('neg数量：{} pos数量：{} part数量:{}'.format(len(negative_list), len(pos_keep), len(part_keep)))

    create_h5_box(positive_list, filename=save_dir + '/positive.h5')
    create_h5_box(part_list, filename=save_dir + '/part.h5')
    create_h5_img(negative_list, filename=save_dir + '/negative.h5')


if __name__ == '__main__':
    weight_dir = '../saved_models'
    mode = 1

    if mode == 1:
        size = 24
    elif mode == 2:
        size = 48
    out_dir = '../data/Face/{}'.format(size)

    dataset = load_widerface_dataset(images_dir=WIDER_FACE_IMG_DIR, anno_file=WIDER_FACE_ANNO_FILE)
    detector = Detector(weight_dir, min_face_size=24, threshold=[0.6, 0.7, 0.7], scale_factor=0.65, mode=mode)
    bboxes_all = []
    # p_net 一次测一张图片，注意，其返回可能会有多个，因为图片中可以包含多张面，而且还有图片金字塔
    print('data img len --:', len(dataset['images']))

    for img_jpg in tqdm(dataset['images']):
        _, bboxes = detector.predict(img_jpg)
        bboxes_all.append(bboxes)

    bboxes_all = np.array(bboxes_all)
    print('predict over---')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    save_hard_example(size, out_dir, dataset, bboxes_all)



#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/11/27
@Description:
"""

import numpy as np
import h5py
import sys
from tqdm import tqdm, trange
import glob
import cv2


def cal_iou(box,boxes):
    '''裁剪的box和图片所有人脸box的iou值
    参数：
      box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
      boxes：图片所有人脸box,[n,4]
    返回值：
      iou值，[n,]
    '''
    #box面积
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    #boxes面积,[n,]
    area=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    #重叠部分左上右下坐标
    xx1=np.maximum(box[0],boxes[:,0])
    yy1=np.maximum(box[1],boxes[:,1])
    xx2=np.minimum(box[2],boxes[:,2])
    yy2=np.minimum(box[3],boxes[:,3])
    
    #重叠部分长宽
    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)
    #重叠部分面积
    inter=w*h
    return inter/(box_area+area-inter+1e-10)


def read_annotation(base_dir, label_path):
    '''读取文件的image，box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')
     
        one_image_bboxes = []
        for i in range(int(nums)):
           
            bb_info = labelfile.readline().strip('\n').split(' ')
            #人脸框
            face_box = [float(bb_info[i]) for i in range(4)]
            
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
           
        bboxes.append(one_image_bboxes)

    data['images'] = images
    data['bboxes'] = bboxes
    return data


def convert_to_square(box):
    '''将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    '''
    square_box=box.copy()
    h=box[:,3]-box[:,1]+1
    w=box[:,2]-box[:,0]+1
    #找寻正方形最大边长
    max_side=np.maximum(w,h)
    
    square_box[:,0]=box[:,0]+w*0.5-max_side*0.5
    square_box[:,1]=box[:,1]+h*0.5-max_side*0.5
    square_box[:,2]=square_box[:,0]+max_side-1
    square_box[:,3]=square_box[:,1]+max_side-1
    return square_box


def create_h5_lmk(data, filename='landmark.h5'):
    size = len(data)
    image, landmark = data[:, 0], data[:, 1]

    hdf5_dataset = h5py.File(filename, 'w')
    hdf5_images = hdf5_dataset.create_dataset(name='images',
                                              shape=(size,),
                                              maxshape=(None),
                                              dtype=h5py.special_dtype(vlen=np.float32))
    hdf5_landmarks = hdf5_dataset.create_dataset(name='landmarks',
                                                 shape=(size,),
                                                 maxshape=(None),
                                                 dtype=h5py.special_dtype(vlen=np.float32))

    tr = trange(size, desc='Creating ' + filename, file=sys.stdout)
    for idx in tr:
        img = (image[idx] - 127.5) / 128.
        hdf5_images[idx] = img.reshape(-1)
        hdf5_landmarks[idx] = landmark[idx]

    hdf5_dataset.close()


def read_h5_lmk(filename, shape):
    hdf5_dataset = h5py.File(filename, 'r')
    dataset_size = len(hdf5_dataset['images'])

    tr = trange(dataset_size, desc='Loading ' + filename, file=sys.stdout)
    images, landmarks = [], []
    for idx in tr:
        image = hdf5_dataset['images'][idx].reshape(shape)
        images.append(image)
        landmarks.append(hdf5_dataset['landmarks'][idx])

    hdf5_dataset.close()
    return np.asarray(images), np.asarray(landmarks)


def create_h5_box(data, filename='positive.h5'):
    size = len(data)
    image, box = data[:, 0], data[:, 1]
    hdf5_dataset = h5py.File(filename, 'w')
    hdf5_images = hdf5_dataset.create_dataset(name='images',
                                              shape=(size,),
                                              maxshape=(None),
                                              dtype=h5py.special_dtype(vlen=np.float32))
    hdf5_boxes = hdf5_dataset.create_dataset(name='boxes',
                                             shape=(size,),
                                             maxshape=(None),
                                             dtype=h5py.special_dtype(vlen=np.float32))
    tr = trange(size, desc='Creating ' + filename, file=sys.stdout)
    for idx in tr:
        img = (image[idx] - 127.5) / 128.
        hdf5_images[idx] = img.reshape(-1)
        hdf5_boxes[idx] = box[idx]

    hdf5_dataset.close()


def read_h5_box(filename, shape):
    hdf5_dataset = h5py.File(filename, 'r')
    dataset_size = len(hdf5_dataset['images'])

    tr = trange(dataset_size, desc='Loading ' + filename, file=sys.stdout)
    images, boxes = [], []
    for idx in tr:
        image = hdf5_dataset['images'][idx].reshape(shape)
        images.append(image)
        boxes.append(hdf5_dataset['boxes'][idx])

    hdf5_dataset.close()
    return np.asarray(images), np.asarray(boxes)


def create_h5_img(data, filename='negative.h5'):
    size = len(data)
    image = data
    hdf5_dataset = h5py.File(filename, 'w')
    hdf5_images = hdf5_dataset.create_dataset(name='images',
                                              shape=(size,),
                                              maxshape=(None),
                                              dtype=h5py.special_dtype(vlen=np.float32))

    tr = trange(size, desc='Creating ' + filename, file=sys.stdout)
    for idx in tr:
        img = (image[idx] - 127.5) / 128.
        hdf5_images[idx] = img.reshape(-1)

    hdf5_dataset.close()


def read_h5_img(filename, shape):
    hdf5_dataset = h5py.File(filename, 'r')
    dataset_size = len(hdf5_dataset['images'])

    tr = trange(dataset_size, desc='Loading ' + filename, file=sys.stdout)
    images, boxes = [], []
    for idx in tr:
        image = hdf5_dataset['images'][idx].reshape(shape)
        images.append(image)

    hdf5_dataset.close()
    return np.asarray(images)


def create_h5_datasets(data, filename='positive.h5'):
    size = len(data)
    image, label, box = data[:, 0], data[:, 1], data[:, 2]

    hdf5_dataset = h5py.File(filename, 'w')
    hdf5_images = hdf5_dataset.create_dataset(name='images',
                                              shape=(size, ),
                                              maxshape=(None),
                                              dtype=h5py.special_dtype(vlen=np.uint8))
    hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                              shape=(size, ),
                                              maxshape=(None),
                                              dtype=h5py.special_dtype(vlen=np.int8))
    hdf5_boxes = hdf5_dataset.create_dataset(name='boxes',
                                             shape=(size, ),
                                             maxshape=(None),
                                             dtype=h5py.special_dtype(vlen=np.float32))

    tr = trange(size, desc='Creating '+filename, file=sys.stdout)
    for idx in tr:
        hdf5_images[idx] = image[idx].reshape(-1)
        hdf5_labels[idx] = label[idx]
        hdf5_boxes[idx] = box[idx]

    hdf5_dataset.close()


def read_h5_datasets(filename, shape):
    hdf5_dataset = h5py.File(filename, 'r')
    dataset_size = len(hdf5_dataset['images'])

    tr = trange(dataset_size, desc='Loading ' + filename, file=sys.stdout)
    images, labels, boxes = [], [], []
    for idx in tr:
        image = hdf5_dataset['images'][idx].reshape(shape)
        images.append(image)
        labels.append(hdf5_dataset['labels'][idx])
        boxes.append(hdf5_dataset['boxes'][idx])
    hdf5_dataset.close()
    return np.asarray(images), np.asarray(labels), np.asarray(boxes)


class DataLoader(object):
    def __init__(self, pos, part, neg, lmk=None, batch_size=5):
        if lmk is None:
            pos_ratio, part_ratio, neg_ratio = 1./5., 1./5., 3./5.

            self.pos_images, self.pos_boxes = pos
            self.part_images, self.part_boxes = part
            self.neg_images = neg

            self.pos_size, self.part_size, self.neg_size =\
                int(batch_size*pos_ratio), int(batch_size*part_ratio), int(batch_size*neg_ratio)

            self.new_batch_size = int(self.pos_size+self.part_size+self.neg_size)
            self.sum_batch = int(min(len(self.pos_images) // self.pos_size,
                                     len(self.part_images) // self.part_size,
                                     len(self.neg_images) // self.neg_size))

        else:
            pos_ratio, part_ratio, neg_ratio, lmk_ratio = 1./6., 1./6., 3./6., 1./6.

            self.pos_images, self.pos_boxes = pos
            self.part_images, self.part_boxes = part
            self.neg_images = neg
            self.lmk_images, self.lmk_landmarks = lmk

            self.pos_size, self.part_size, self.neg_size, self.lmk_size = \
                int(batch_size * pos_ratio), int(batch_size * part_ratio), \
                int(batch_size * neg_ratio), int(batch_size * lmk_ratio)

            self.new_batch_size = int(self.pos_size + self.part_size + self.neg_size)
            self.sum_batch = int(min(len(self.pos_images) // self.pos_size,
                                     len(self.part_images) // self.part_size,
                                     len(self.neg_images) // self.neg_size,
                                     len(self.lmk_images) // self.lmk_size))

        print("new batch size", self.new_batch_size)
        print("sum_batch", self.sum_batch)

    def get_step(self):
        return self.sum_batch

    def generate(self):
        pos_label = np.array([[1]]).repeat(self.pos_size, axis=0)
        part_label = np.array([[-1]]).repeat(self.part_size, axis=0)
        neg_label = np.array([[0]]).repeat(self.neg_size, axis=0)
        neg_box = np.array([[-1., -1., -1., -1.]]).repeat(self.neg_size, axis=0)

        current_batch = 0

        while True:
            if current_batch >= self.sum_batch-1:
                current_batch = 0
            image_batch, label_batch, box_batch = [], [], []
            start = int(current_batch * self.pos_size)
            end = int((current_batch+1)*self.pos_size)
            image_batch.extend(self.pos_images[start:end])
            label_batch.extend(pos_label)
            box_batch.extend(self.pos_boxes[start:end])

            start = int(current_batch * self.part_size)
            end = int((current_batch + 1) * self.part_size)
            image_batch.extend(self.part_images[start:end])
            label_batch.extend(part_label)
            box_batch.extend(self.part_boxes[start:end])

            start = int(current_batch * self.neg_size)
            end = int((current_batch + 1) * self.neg_size)
            image_batch.extend(self.neg_images[start:end])
            label_batch.extend(neg_label)
            box_batch.extend(neg_box)

            # target_batch = []
            # for i in range(self.new_batch_size):
            #     target_batch.append(np.concatenate((label_batch[i], box_batch[i])))
            target_batch = np.concatenate((label_batch, box_batch), axis=-1)
            target_batch = np.asarray(target_batch).astype(np.float32)
            image_batch = np.asarray(image_batch).astype(np.float32)

            current_batch += 1

            yield image_batch, target_batch

    def generate_lmk(self):
        pos_label = np.array([[1]]).repeat(self.pos_size, axis=0)
        pos_landmark = np.zeros((1, 10)).repeat(self.pos_size, axis=0)
        part_label = np.array([[-1]]).repeat(self.part_size, axis=0)
        part_landmark = np.zeros((1, 10)).repeat(self.part_size, axis=0)
        neg_label = np.array([[0]]).repeat(self.neg_size, axis=0)
        neg_box = np.zeros((1, 4)).repeat(self.neg_size, axis=0)
        neg_landmark = np.zeros((1, 10)).repeat(self.neg_size, axis=0)
        lmk_label = np.array([[-2]]).repeat(self.lmk_size, axis=0)
        lmk_box = np.zeros((1, 4)).repeat(self.lmk_size, axis=0)
        current_batch = 0

        while True:
            if current_batch >= self.sum_batch - 1:
                current_batch = 0
            image_batch, label_batch, box_batch, landmark_batch = [], [], [], []
            start = int(current_batch * self.pos_size)
            end = int((current_batch + 1) * self.pos_size)
            image_batch.extend(self.pos_images[start:end])
            label_batch.extend(pos_label)
            box_batch.extend(self.pos_boxes[start:end])
            landmark_batch.extend(pos_landmark)

            start = int(current_batch * self.part_size)
            end = int((current_batch + 1) * self.part_size)
            image_batch.extend(self.part_images[start:end])
            label_batch.extend(part_label)
            box_batch.extend(self.part_boxes[start:end])
            landmark_batch.extend(part_landmark)

            start = int(current_batch * self.neg_size)
            end = int((current_batch + 1) * self.neg_size)
            image_batch.extend(self.neg_images[start:end])
            label_batch.extend(neg_label)
            box_batch.extend(neg_box)
            landmark_batch.extend(neg_landmark)

            start = int(current_batch * self.lmk_size)
            end = int((current_batch + 1) * self.lmk_size)
            image_batch.extend(self.lmk_images[start:end])
            label_batch.extend(lmk_label)
            box_batch.extend(lmk_box)
            landmark_batch.extend(self.lmk_landmarks[start:end])

            # target_batch = []
            # for i in range(self.new_batch_size):
            #     target_batch.append(np.concatenate((label_batch[i], box_batch[i])))
            target_batch = np.concatenate((label_batch, box_batch, landmark_batch), axis=-1)
            target_batch = np.asarray(target_batch).astype(np.float32)
            image_batch = np.asarray(image_batch).astype(np.float32)

            current_batch += 1
            # print(image_batch.shape)
            # print(target_batch.shape)
            yield image_batch, target_batch


def py_nms(bboxes, thresh, mode="union"):
    assert mode in ['union', 'minimum']

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        else:
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def process_image(img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        return img_resized


def convert_to_square(bbox):
    # print('bbox_2_square---:', bbox.shape)
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

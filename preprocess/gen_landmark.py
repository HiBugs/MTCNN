#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/11/28
@Description:
"""

import os
import random
import sys
import cv2
import numpy as np
npr=np.random
import argparse
from tqdm import tqdm
from utils import cal_iou, create_h5_lmk
from bbox_utils import get_landmark, BBox


size = 48
data_dir = '../data'
save_dir = '../data/Face/{}'.format(size)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def flip(face,landmark):
    #镜像
    face_flipped_by_x=cv2.flip(face,1)
    landmark_=np.asarray([(1-x,y) for (x,y) in landmark])
    landmark_[[0,1]]=landmark_[[1,0]]
    landmark_[[3,4]]=landmark_[[4,3]]
    return (face_flipped_by_x,landmark_)


def rotate(img,box,landmark,alpha):
    #旋转
    center=((box.left+box.right)/2,(box.top+box.bottom)/2)
    rot_mat=cv2.getRotationMatrix2D(center,alpha,1)
    img_rotated_by_alpha=cv2.warpAffine(img,rot_mat,(img.shape[1],img.shape[0]))
    landmark_=np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                         rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x,y) in landmark ])
    face=img_rotated_by_alpha[box.top:box.bottom+1,box.left:box.right+1]
    return (face,landmark_)


def create_landmark(argument=True):
    # 是否对图像变换
    # argument = True
    image_id = 0

    ftxt = os.path.join(data_dir, 'trainImageList.txt')
    data = get_landmark(ftxt, data_dir)
    idx = 0
    landmark_list = []

    for (imgPath,box,landmarkGt) in tqdm(data):
        # if image_id > 3:
        #     break
        #存储人脸图片和关键点
        F_imgs=[]
        F_landmarks=[]
        img=cv2.imread(imgPath, cv2.COLOR_BGR2RGB)

        img_h,img_w,img_c=img.shape
        gt_box=np.array([box.left,box.top,box.right,box.bottom])
        #人脸图片
        f_face=img[box.top:box.bottom+1,box.left:box.right+1]
        #resize成网络输入大小
        f_face=cv2.resize(f_face,(size,size))

        landmark=np.zeros((5,2))
        for index ,one in enumerate(landmarkGt):
            #关键点相对于左上坐标偏移量并归一化
            rv=((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]),(one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index]=rv
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        landmark=np.zeros((5,2))
        if argument:
            #对图像变换
            idx=idx+1
            x1,y1,x2,y2=gt_box
            gt_w=x2-x1+1
            gt_h=y2-y1+1
            #除去过小图像
            if max(gt_w,gt_h)<40 or x1<0 or y1<0:
                continue
            for i in range(10):
                #随机裁剪图像大小
                box_size=npr.randint(int(min(gt_w,gt_h)*0.8),np.ceil(1.25*max(gt_w,gt_h)))
                #随机左上坐标偏移量
                delta_x=npr.randint(-gt_w*0.2,gt_w*0.2)
                delta_y=npr.randint(-gt_h*0.2,gt_h*0.2)
                #计算左上坐标
                nx1=int(max(x1+gt_w/2-box_size/2+delta_x,0))
                ny1=int(max(y1+gt_h/2-box_size/2+delta_y,0))
                nx2=nx1+box_size
                ny2=ny1+box_size
                #除去超过边界的
                if nx2>img_w or ny2>img_h:
                    continue
                #裁剪边框，图片
                crop_box=np.array([nx1,ny1,nx2,ny2])
                cropped_im=img[ny1:ny2+1,nx1:nx2+1,:]
                resized_im=cv2.resize(cropped_im,(size,size))
                iou=cal_iou(crop_box,np.expand_dims(gt_box,0))
                #只保留pos图像
                if iou>0.65:
                    F_imgs.append(resized_im)
                    #关键点相对偏移
                    for index,one in enumerate(landmarkGt):
                        rv=((one[0]-nx1)/box_size,(one[1]-ny1)/box_size)
                        landmark[index]=rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark=np.zeros((5,2))
                    landmark_=F_landmarks[-1].reshape(-1,2)
                    box=BBox([nx1,ny1,nx2,ny2])
                    #镜像
                    if random.choice([0,1])>0:
                        face_flipped,landmark_flipped=flip(resized_im,landmark_)
                        face_flipped=cv2.resize(face_flipped,(size,size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    #逆时针翻转
                    if random.choice([0,1])>0:
                        face_rotated_by_alpha,landmark_rorated=rotate(img,box,box.reprojectLandmark(landmark_),5)
                        #关键点偏移
                        landmark_rorated=box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha=cv2.resize(face_rotated_by_alpha,(size,size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))

                        #左右翻转
                        face_flipped,landmark_flipped=flip(face_rotated_by_alpha,landmark_rorated)
                        face_flipped=cv2.resize(face_flipped,(size,size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    #顺时针翻转
                    if random.choice([0,1])>0:
                        face_rotated_by_alpha,landmark_rorated=rotate(img,box,box.reprojectLandmark(landmark_),-5)
                        #关键点偏移
                        landmark_rorated=box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha=cv2.resize(face_rotated_by_alpha,(size,size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))

                        #左右翻转
                        face_flipped,landmark_flipped=flip(face_rotated_by_alpha,landmark_rorated)
                        face_flipped=cv2.resize(face_flipped,(size,size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

        for i in range(len(F_imgs)):
            #剔除数据偏移量在[0,1]之间
            if np.sum(np.where(F_landmarks[i]<=0,1,0))>0:
                continue
            if np.sum(np.where(F_landmarks[i]>=1,1,0))>0:
                continue
            landmark_list.append([F_imgs[i],F_landmarks[i]])

            image_id += 1

    print("landmark数量：", image_id)
    return landmark_list


if __name__ == '__main__':
    base_num = 10000
    landmark_list = create_landmark(argument=True)
    landmark_list = np.asarray(landmark_list)
    if len(landmark_list) > base_num:
        landmark_keep = npr.choice(len(landmark_list), base_num, replace=True)
        landmark_list = landmark_list[landmark_keep]

    create_h5_lmk(landmark_list, filename=save_dir+'/landmark.h5')

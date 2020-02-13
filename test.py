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
from detector import Detector
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


img_path = "images/03.jpg"
weight_dir = 'saved_models'
mode = 3
detector = Detector(weight_dir, min_face_size=24, threshold=[0.6, 0.7, 0.7], scale_factor=0.65, mode=mode)

image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
result = detector.predict(image)    # _, bboxes, landmark
bboxes = result[1]
print('bboxes-shape---:', bboxes.shape)
for bbox in bboxes:
    cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(255,0,255))
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))

if mode == 3:
    landmarks = result[2]
    for landmark in landmarks:
        for i in range(0, 5):
            cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 2, (0,0,255))

cv2.imwrite('result.png', image)
cv2.imshow('detect', image)
cv2.waitKey(0)

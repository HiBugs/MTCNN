#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/11/27
@Description:
"""
import os
import numpy as np

from nets.mtcnn import o_net
from nets.losses import mt_lmk_loss, cal_accuracy
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from preprocess.utils import DataLoader, read_h5_img, read_h5_box, read_h5_lmk
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def lr_schedule(epoch):
    if epoch < 50:
        return 0.01
    elif epoch < 200:
        return 0.001
    elif epoch < 500:
        return 0.0001
    elif epoch < 700:
        return 0.00005
    elif epoch < 900:
        return 0.000005
    else:
        return 0.000001


batch_size = 64*10
pos = read_h5_box("./data/Face/48/positive.h5", (48, 48, 3))
part = read_h5_box("./data/Face/48/part.h5", (48, 48, 3))
neg = read_h5_img("./data/Face/48/negative.h5", (48, 48, 3))
lmk = read_h5_lmk("./data/Face/48/landmark.h5", (48, 48, 3))
data = DataLoader(pos, part, neg,  lmk, batch_size=batch_size)
gen = data.generate_lmk()
step = data.get_step()

model = o_net(training=True)
# model.load_weights('saved_models/pnet.h5')
opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=mt_lmk_loss, metrics=[cal_accuracy])
model.summary()

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
csv_logger = CSVLogger('saved_models/onet.log', append=False)
model_checkpoint = ModelCheckpoint('saved_models/onet.h5', monitor='loss', mode='min', verbose=1,
                                   save_weights_only=True, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, learning_rate_scheduler]

model.fit_generator(gen, epochs=1000, verbose=1, callbacks=callbacks,
                    steps_per_epoch=step)

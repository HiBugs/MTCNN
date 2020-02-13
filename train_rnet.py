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

from nets.mtcnn import r_net
from nets.losses import mt_loss, cal_accuracy
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from preprocess.utils import DataLoader, read_h5_img, read_h5_box
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def lr_schedule(epoch):
    if epoch < 50:
        return 0.01
    elif epoch < 300:
        return 0.001
    elif epoch < 600:
        return 0.0001
    else:
        return 0.00005


batch_size = 64*10
pos = read_h5_box("./data/Face/24/positive.h5", (24, 24, 3))
part = read_h5_box("./data/Face/24/part.h5", (24, 24, 3))
neg = read_h5_img("./data/Face/24/negative.h5", (24, 24, 3))
data = DataLoader(pos, part, neg,  batch_size=batch_size)
gen = data.generate()
step = data.get_step()

model = r_net(training=True)
# model.load_weights('saved_models/pnet.h5')
opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=mt_loss, metrics=[cal_accuracy])
model.summary()

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
csv_logger = CSVLogger('saved_models/rnet.log', append=False)
model_checkpoint = ModelCheckpoint('saved_models/rnet.h5', monitor='loss', mode='min', verbose=1,
                                   save_weights_only=True, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, learning_rate_scheduler]

model.fit_generator(gen, epochs=1000, verbose=1, callbacks=callbacks,
                    steps_per_epoch=step)

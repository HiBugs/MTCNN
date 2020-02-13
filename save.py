#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/12/1
@Description:
"""
from nets.mtcnn import p_net, r_net
from tensorflow.keras.models import save_model

model = p_net()
model.load_weights('saved_models/pnet.h5')
save_model(model, 'saved_models/PNET.h5')
print("----------- p_net saved! -----------")

# from tensorflow import lite
# converter = lite.TFLiteConverter.from_keras_model_file('saved_models/P_NET.h5')
# tflite_model = converter.convert()
# with open("p_net.tflite", "wb") as f:
#     f.write(tflite_model)

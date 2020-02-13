#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/11/27
@Description:
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, PReLU, MaxPooling2D, Dense
from tensorflow.keras.layers import Input, Reshape, Concatenate, Flatten


def p_net(training=False):
    x = Input(shape=(12, 12, 3)) if training else Input(shape=(None, None, 3))
    y = Conv2D(10, 3, padding='valid', strides=(1, 1), kernel_initializer='he_normal', name='p_conv1')(x)
    y = PReLU(shared_axes=(1, 2), name='p_prelu1')(y)
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='p_max_pooling1')(y)
    y = Conv2D(16, 3, padding='valid', strides=(1, 1), kernel_initializer='he_normal', name='p_conv2')(y)
    y = PReLU(shared_axes=(1, 2), name='p_prelu2')(y)
    y = Conv2D(32, 3, padding='valid', strides=(1, 1), kernel_initializer='he_normal', name='p_conv3')(y)
    y = PReLU(shared_axes=(1, 2), name='p_prelu3')(y)

    classifier = Conv2D(2, 1, activation='softmax', kernel_initializer='he_normal', name='p_conv_cls')(y)
    bbox = Conv2D(4, 1, kernel_initializer='he_normal', name='p_conv_bbox')(y)

    if training:
        classifier = Reshape((2,), name='p_cls')(classifier)
        bbox = Reshape((4,), name='p_bbox')(bbox)
        outputs = Concatenate(name='p_predict')([classifier, bbox])
        model = Model(inputs=[x], outputs=[outputs], name='P_Net')
    else:
        model = Model(inputs=[x], outputs=[classifier, bbox], name='P_Net')
    return model


def r_net(training=False):
    x = Input(shape=(24, 24, 3))
    y = Conv2D(28, 3, padding='same', strides=(1, 1), name='r_conv1')(x)
    y = PReLU(shared_axes=(1, 2), name='r_prelu1')(y)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='p_max_pooling1')(y)
    y = Conv2D(48, 3, padding='valid', strides=(1, 1), name='r_conv2')(y)
    y = PReLU(shared_axes=(1, 2), name='r_prelu2')(y)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='p_max_pooling2')(y)
    y = Conv2D(64, 2, padding='valid', name='r_conv3')(y)
    y = PReLU(shared_axes=(1, 2), name='r_prelu3')(y)
    y = Dense(128, name='r_dense')(y)
    y = PReLU(name='r_prelu4')(y)
    y = Flatten()(y)

    classifier = Dense(2, activation='softmax', name='r_classifier')(y)
    bbox = Dense(4, name='r_bbox')(y)

    if training:
        outputs = Concatenate(name='r_predict')([classifier, bbox])
        model = Model(inputs=[x], outputs=[outputs], name='R_Net')
    else:
        model = Model(inputs=[x], outputs=[classifier, bbox], name='R_Net')

    return model


def o_net(training=False):
    x = Input(shape=(48, 48, 3))
    y = Conv2D(32, 3, padding='same', strides=(1, 1), name='o_conv1')(x)
    y = PReLU(shared_axes=(1, 2), name='o_prelu1')(y)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='o_max_pooling1')(y)
    y = Conv2D(64, 3, padding='valid', strides=(1, 1), name='o_conv2')(y)
    y = PReLU(shared_axes=(1, 2), name='o_prelu2')(y)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='o_max_pooling2')(y)
    y = Conv2D(64, 3, padding='valid', strides=(1, 1), name='o_conv3')(y)
    y = PReLU(shared_axes=(1, 2), name='o_prelu3')(y)
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='o_max_pooling3')(y)
    y = Conv2D(128, 2, padding='valid', strides=(1, 1), name='o_conv4')(y)
    y = PReLU(shared_axes=(1, 2), name='o_prelu4')(y)
    y = Dense(256, name='o_dense')(y)
    y = PReLU(name='o_prelu5')(y)
    y = Flatten()(y)

    classifier = Dense(2, activation='softmax', name='o_classifier')(y)
    bbox = Dense(4, name='o_bbox')(y)
    landmark = Dense(10, name='o_landmark')(y)

    if training:
        outputs = Concatenate(name='o_predict')([classifier, bbox, landmark])
        model = Model(inputs=[x], outputs=[outputs], name='O_Net')
    else:
        model = Model(inputs=[x], outputs=[classifier, bbox, landmark], name='O_Net')
    return model


if __name__ == '__main__':
    p = p_net(training=True)
    p.summary()

    # r = r_net()
    # r.summary()
    #
    # o = o_net()
    # o.summary()
